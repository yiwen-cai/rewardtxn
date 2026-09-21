import dataclasses,importlib.util,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'tests/ft'))
from scripts.ft import areal_ft1 as entry
import run_ft1_faults as faults
from test_ft1_preflight import Preparation

class FaultPreparation(unittest.TestCase):
 def test_fault_configs_use_same_async_and_native_retry(self):
  from areal.api.cli_args import GRPOConfig,load_expr_config
  import tempfile
  base=(repo/'docs/experiments/rewardtxn-ft-20260916/native-trainer.yaml').read_text().replace('async_save: false','async_save: true')
  for scenario,(seed,order) in faults.CASES.items():
   with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'config.yaml';path.write_text(faults.fault_config(base,scenario,seed))
    config,_=load_expr_config(['--config',str(path)],GRPOConfig)
    self.assertEqual((config.seed,config.total_train_steps,config.recover.retries),(seed,10,1))
    self.assertTrue(config.actor.megatron.async_save)
    self.assertEqual((config.gconfig.n_samples,config.train_dataset.batch_size),(8,4))
   with self.assertRaises(ValueError):faults.fault_config(base,scenario,211)
 def test_save_load_observer_preserves_calls_and_records_state(self):
  from areal.engine.megatron_engine import MegatronPPOActor
  from areal.utils.recover import RecoverInfo
  import scripts.ft.training_adapter as adapter
  @dataclasses.dataclass
  class Step:
   global_step:int=2
  calls=[];records=[]
  def save(self,meta):calls.append(('save',meta));return 123
  def load(self,meta):calls.append(('load',meta));return 456
  @classmethod
  def recover(cls,path):return SimpleNamespace(last_step_info=Step())
  with patch.object(MegatronPPOActor,'save',save),patch.object(MegatronPPOActor,'load',load),patch.object(RecoverInfo,'load',recover),patch.object(adapter,'native_snapshot',lambda actor:{'fixture':True}),patch.object(entry,'observe',lambda event,**fields:records.append((event,fields))):
   entry.install_load_observer()
   actor=SimpleNamespace(_pilot_update='u');meta=SimpleNamespace(path='/checkpoint')
   self.assertEqual(MegatronPPOActor.save(actor,meta),123)
   self.assertEqual(MegatronPPOActor.load(actor,meta),456)
   self.assertEqual(RecoverInfo.load('/metadata').last_step_info.global_step,2)
   self.assertEqual(calls,[('save',meta),('load',meta)])
   self.assertEqual([x[0] for x in records],['checkpoint_save_state','checkpoint_loaded_state','recover_info_loaded'])
   self.assertEqual(records[0][1]['state'],records[1][1]['state'])

if __name__=='__main__':unittest.main(verbosity=2)
