"""Actual pinned source contracts, not a fabricated exl3 module API."""
from pathlib import Path
import importlib.util
import unittest

ROOT=Path(__file__).resolve().parents[1]
file=ROOT/'overlay/usr/local/lib/python3.12/dist-packages/hybrid/prefill_api_contract.py'
spec=importlib.util.spec_from_file_location('actual_prefill_api',file)
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
SITE=ROOT/'overlay/usr/local/lib/python3.12/dist-packages'

class ActualAPITests(unittest.TestCase):
    def test_actual_mounted_source_copies_and_full_api_pass(self):
        result=api.audit(ROOT)
        self.assertTrue(result['passed'])
        self.assertEqual(result['checks']['plugin']['hook_global'],'_tc_hook')
        self.assertEqual(result['checks']['hook']['workspace_dtype'],'float32')

    def test_previous_wrong_donor_lookup_is_rejected(self):
        source=(SITE/'hybrid/prefill_startup.py').read_text(encoding='utf-8')
        source=source.replace('hook=exl3._tc_hook','hook=exl3._ds41_tc_hook()')
        with self.assertRaisesRegex(ValueError,'hook lookup'):api.check_startup(source)

    def test_actual_hook_signature_and_keys_are_enforced(self):
        source=(ROOT/'prefill-reference/live-api/exl3_tc_hook.py').read_text(encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'signature'):
            api.check_hook(source.replace('def try_apply(x2d, ids, weights, layer, expert_map, limit):','def try_apply(x2d, ids, weights, layer, inners, expert_map, limit):'))
        with self.assertRaisesRegex(ValueError,'TC keys'):
            api.check_hook(source.replace('"workspace": workspace,','"wrong_workspace": workspace,'))

    def test_actual_workspace_dtype_is_enforced(self):
        source=(ROOT/'prefill-reference/live-api/exl3_tc_hook.py').read_text(encoding='utf-8')
        bad=source.replace('workspace = torch.empty(max(n_v2, n_pre), dtype=torch.float32, device=device)',
                           'workspace = torch.empty(max(n_v2, n_pre), dtype=torch.float16, device=device)')
        with self.assertRaisesRegex(ValueError,'workspace must be FP32'):api.check_hook(bad)

    def test_actual_plugin_callsite_is_enforced(self):
        source=(ROOT/'prefill-reference/live-api/production-plugin.py').read_text(encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'callsite'):
            api.check_plugin(source.replace('_tc_hook.try_apply(x2d, ids, weights, layer, expert_map, limit)',
                                            '_tc_hook.try_apply(x2d, ids, weights, layer, limit, expert_map)'))

    def test_candidate_abi_order_is_enforced(self):
        source=(SITE/'hybrid/prefill_runtime.py').read_text(encoding='utf-8')
        native=(ROOT/'prefill-native/moe_prefill_try.cu').read_text(encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'ABI'):
            api.check_candidate(source.replace('tc["bits_x2"], out, tc["workspace"], float(limit or 0.0)',
                                              'tc["bits_x2"], float(limit or 0.0), out, tc["workspace"]'),native)

if __name__=='__main__':unittest.main(verbosity=2)
