"""Docker entry point: speed by default, explicit BF16/quality boot presets."""
import argparse,json,os
from lanes import config,environment,validate_command
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--lane',choices=('speed','bf16','quality'),default=os.environ.get('DS1017_LANE','speed'))
 ap.add_argument('action',nargs='?',choices=('describe','serve'),default='describe');ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
 c=config(a.lane)
 if a.action=='describe':
  if a.command:ap.error('Put --lane before describe; describe accepts no server command')
  print(json.dumps(dict(default_lane='speed',selected=c,cold_load_for_projection_change=True,quality_available_in_either_bank=True,gpu_qualification='Required at each cold load by the retained all-rank startup gates; no new measured speed is implied by packaging.'),indent=2));return
 command=a.command[1:] if a.command[:1]==['--'] else a.command;validate_command(command)
 env=environment(a.lane,os.environ);os.execvpe(command[0],command,env)
if __name__=='__main__':main()
