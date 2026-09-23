"""Read-only input checkpoint verification in a fresh single-GPU process."""
import argparse
import dataclasses
import json
import os
from pathlib import Path


def gpu_id(value):
    value=str(value)
    return value[4:] if value.startswith('GPU-') else value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--arm',choices=('A','R'),required=True)
    args=parser.parse_args()
    root,out=Path(args.input),Path(args.output)
    from areal.api.cli_args import GRPOConfig,load_expr_config
    from areal.api import FinetuneSpec,SaveLoadMeta
    from areal.api.alloc_mode import ModelAllocation
    from areal.engine.megatron_engine import MegatronPPOActor
    from areal.utils.seeding import set_random_seed
    from areal.utils.dataloader import create_dataloader
    from areal.utils.recover import RecoverInfo
    from scripts.ft.areal_pilot import load_pilot_dataset
    from scripts.ft.training_adapter import native_snapshot,normalized
    cfg,_=load_expr_config(['--config',str(root/'training.yaml')],GRPOConfig)
    import torch
    assert torch.cuda.device_count()==1
    actual=str(torch.cuda.get_device_properties(0).uuid)
    assert gpu_id(actual)==gpu_id(os.environ['FT1_GPU_UUID'])
    set_random_seed(cfg.seed,'ft1-trainer')
    loader=create_dataloader(load_pilot_dataset(cfg.train_dataset.path),rank=0,world_size=1,dataset_config=cfg.train_dataset)
    saved_reference=None
    if args.arm=='A':
        base=root/'areal/checkpoints/caiyiwen'/cfg.experiment_name/cfg.trial_name
        checkpoint=base/'default/recover_checkpoint'
        recover=base/'recover_info'
    else:
        control=json.loads((root/'rewardtxn/state/control.json').read_text())
        generation=root/'rewardtxn/state/generations'/control['head']['generation']/'checkpoint'
        checkpoint=generation/'native'
        recover=generation/'recover'
        saved_reference=json.loads((generation/'native-state.json').read_text())
    terminal=json.loads((root/'final-native-state.json').read_text())
    engine=MegatronPPOActor(cfg.actor)
    report={'arm':args.arm,'checkpoint':str(checkpoint),'pid':os.getpid(),
            'scope':'fresh process native engine load and RecoverInfo; no training or source ledger mutation'}
    try:
        engine.create_process_group(ModelAllocation.from_str(cfg.actor.backend).parallel)
        engine.initialize(addr=None,ft_spec=FinetuneSpec(total_train_epochs=cfg.total_train_epochs,
            dataset_size=len(loader)*cfg.train_dataset.batch_size,train_batch_size=cfg.train_dataset.batch_size),role='actor')
        engine.load(SaveLoadMeta(path=str(checkpoint),weight_format='dcp',with_optim=True))
        loaded=native_snapshot(engine)
        (out/'loaded-native-state.json').write_text(json.dumps(loaded,sort_keys=True))
        report['terminal_components_equal']={key:loaded[key]==terminal[key] for key in terminal}
        if saved_reference is not None:
            report['saved_components_equal']={key:loaded[key]==saved_reference[key] for key in saved_reference}
        info=RecoverInfo.load(str(recover))
        report['last_step_info']=dataclasses.asdict(info.last_step_info)
        report['next_step_info']=dataclasses.asdict(info.last_step_info.next())
        assert info.last_step_info.global_step==9
        descriptor=info.dataloader_info[0] if isinstance(info.dataloader_info,list) else info.dataloader_info
        if args.arm=='A':
            loader.load_state_dict(descriptor)
            report['native_cursor_roundtrip_equal']=normalized(loader.state_dict())==normalized(descriptor)
        else:
            import hashlib,shutil
            from scripts.ft.replay import DrawLoader
            from scripts.ft.rlvr_replay import digest
            shutil.copytree(root/'rewardtxn/draw',out/'draw-copy')
            source=hashlib.sha256(Path(cfg.train_dataset.path).read_bytes()).hexdigest()
            with DrawLoader(loader,out/'draw-copy',digest(dataclasses.asdict(cfg)),source,
                            digest(dataclasses.asdict(cfg.train_dataset)),k=cfg.gconfig.n_samples) as draw:
                draw.load_state_dict(descriptor)
                restored=draw.state_dict()
            assert restored['wal_sequence']>=descriptor['wal_sequence']
            report['draw_saved_prefix_loaded']=True
            report['saved_wal_sequence']=descriptor['wal_sequence']
            report['restored_wal_sequence']=restored['wal_sequence']
            report['retained_cursor_status']='real DrawLoader restore on copy; consumption independently checked against token chain'
        report['engine_exact_match']=all(report['terminal_components_equal'].values())
        if saved_reference is not None:
            report['engine_exact_match'] &= all(report['saved_components_equal'].values())
        (out/'load-verification.json').write_text(json.dumps(report,indent=2))
    finally:
        engine.destroy()


if __name__=='__main__':main()
