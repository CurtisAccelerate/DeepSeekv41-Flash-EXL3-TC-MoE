"""Record API wall time and actual delivery gaps without changing request settings.

Example:
  python stream_wall_probe.py --url http://192.168.1.252:8095/v1/chat/completions --request request.json --out relay.json

The request JSON must explicitly contain stream=true. Full response events are
saved; credentials are read only from an explicitly named --key-env variable.
"""
import argparse
import codecs
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen


def percentile(values, fraction):
    if not values:
        return None
    a = sorted(values)
    return a[min(len(a) - 1, max(0, int((len(a) - 1) * fraction)))]


def measure(url, body, label='', timeout=3600, api_key=None, extra_headers=None):
    if body.get('stream') is not True:
        raise ValueError('Request must explicitly set stream=true; no other settings are changed')
    payload = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
    if extra_headers:
        headers.update(extra_headers)
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    report = {
        'label': label, 'url': url, 'request_sha256': hashlib.sha256(payload).hexdigest(),
        'request_body': body, 'start_unix_s': time.time(), 'client_chunks': [], 'events': [],
        'content': '', 'reasoning': '', 'usage': None, 'finish_reasons': [],
        'timing_note': 'Client arrival times. Several SSE events may arrive in one read; these are not GPU token timestamps.',
    }
    started = time.perf_counter()
    first_any = first_content = first_reasoning = first_tool = first_byte = done = finish = None
    content_times, raw_times = [], []
    buffered, event_lines = '', []
    decoder = codecs.getincrementaldecoder('utf-8')()

    def event(data, arrived):
        nonlocal first_any, first_content, first_reasoning, first_tool, done, finish
        if data == '[DONE]':
            done = arrived
            report['events'].append({'seconds': arrived, 'done': True})
            return
        try:
            obj = json.loads(data)
        except ValueError:
            report['events'].append({'seconds': arrived, 'unparsed_data': data})
            return
        report['events'].append({'seconds': arrived, 'data': obj})
        if obj.get('id'):
            report['response_id'] = obj['id']
        if obj.get('model'):
            report['response_model'] = obj['model']
        if obj.get('usage'):
            report['usage'] = obj['usage']
        for choice in obj.get('choices') or []:
            delta = choice.get('delta') or {}
            text = delta.get('content') or ''
            reasoning = delta.get('reasoning_content') or delta.get('reasoning') or ''
            tools = delta.get('tool_calls') or delta.get('function_call')
            if text or reasoning or tools:
                if first_any is None:
                    first_any = arrived
            if text:
                if first_content is None:
                    first_content = arrived
                content_times.append(arrived)
                report['content'] += text
            if reasoning:
                if first_reasoning is None:
                    first_reasoning = arrived
                report['reasoning'] += reasoning
            if tools and first_tool is None:
                first_tool = arrived
            if choice.get('finish_reason') is not None:
                finish = arrived
                report['finish_reasons'].append(choice['finish_reason'])

    try:
        with urlopen(Request(url, payload, headers), timeout=timeout) as response:
            report['headers_s'] = time.perf_counter() - started
            report['http_status'] = response.status
            report['response_headers'] = {k: v for k, v in response.headers.items()
                                          if k.lower() in ('content-type', 'server', 'x-request-id')}
            while True:
                chunk = response.read1(4096)
                arrived = time.perf_counter() - started
                if not chunk:
                    break
                if first_byte is None:
                    first_byte = arrived
                raw_times.append(arrived)
                report['client_chunks'].append({'seconds': arrived, 'bytes': len(chunk)})
                buffered += decoder.decode(chunk)
                while '\n' in buffered:
                    line, buffered = buffered.split('\n', 1)
                    line = line.rstrip('\r')
                    if not line:
                        if event_lines:
                            event('\n'.join(event_lines), arrived)
                            event_lines = []
                    elif line.startswith('data:'):
                        event_lines.append(line[5:].lstrip(' '))
            buffered += decoder.decode(b'', final=True)
            if buffered.startswith('data:'):
                event_lines.append(buffered[5:].lstrip(' ').rstrip('\r'))
            if event_lines:
                event('\n'.join(event_lines), time.perf_counter() - started)
    except Exception as exc:
        report['error'] = {'class': type(exc).__name__, 'message': str(exc)}
    wall = time.perf_counter() - started
    gaps = [b - a for a, b in zip(content_times, content_times[1:])]
    raw_gaps = [b - a for a, b in zip(raw_times, raw_times[1:])]
    usage = report['usage'] or {}
    completion = usage.get('completion_tokens')
    report['timing'] = {
        'wall_s': wall, 'first_body_byte_s': first_byte, 'first_generated_event_s': first_any,
        'first_visible_content_s': first_content, 'first_reasoning_s': first_reasoning,
        'first_tool_fragment_s': first_tool, 'finish_event_s': finish, 'done_s': done,
        'last_visible_content_s': content_times[-1] if content_times else None,
        'max_visible_gap_s': max(gaps) if gaps else None,
        'p95_visible_gap_s': percentile(gaps, 0.95),
        'max_body_read_gap_s': max(raw_gaps) if raw_gaps else None,
        'visible_gap_count_over_1s': sum(g > 1 for g in gaps),
        'visible_gap_count_over_5s': sum(g > 5 for g in gaps),
        'visible_event_count': len(content_times),
        'completion_tokens_per_full_wall_s': completion / wall if completion is not None and wall else None,
        'tail_after_done_s': wall - done if done is not None else None,
    }
    report['transport_complete'] = done is not None and 'error' not in report
    report['successful_task_answer'] = None  # Must be judged from the actual requested task.
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', required=True)
    ap.add_argument('--request', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--label', default='')
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--key-env', default=None)
    args = ap.parse_args()
    body = json.loads(args.request.read_text(encoding='utf-8-sig'))
    result = measure(args.url, body, args.label, args.timeout,
                     os.environ[args.key_env] if args.key_env else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'label': result['label'], 'http_status': result.get('http_status'),
                      'timing': result['timing'], 'error': result.get('error')}, indent=2))
    raise SystemExit(0 if result['transport_complete'] else 1)


if __name__ == '__main__':
    main()
