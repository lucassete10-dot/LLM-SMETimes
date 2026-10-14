"""CPU orchestration tests; fixtures are not model accuracy evidence."""
import json,tempfile,unittest
from pathlib import Path
from scripts.optimize_llm_dataset_validation import run_optimization


class OptimizationWorkflowTests(unittest.TestCase):
    def test_sparse_stage_runs_all_four_comparisons_without_claiming_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);queue=self.setup_fixture(root);calls=[]
            def sparse(run):
                file=run/'selected_metrics.json';rows=json.loads(file.read_text(encoding='utf-8'))
                for row in rows.values():row.update(stage_drop_max_pp=None,stage_metrics_available=False)
                file.write_text(json.dumps(rows),encoding='utf-8')
            for task in queue['tasks']:sparse(Path(task['run']))
            base=self.command_fixture(root,calls)
            def command(queue_root,index,arguments,on_run=None):
                log=base(queue_root,index,arguments,on_run)
                if arguments[1]=='scripts.train_llm_curve_repair':
                    sparse(Path(next(line[4:] for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('RUN='))))
                return log
            run_optimization(root,queue,'MIT',command,Path('p0'),Path('p1'),Path('auth'))
            state=json.loads((root/'optimization/MIT/state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['state'],'complete');self.assertIsNone(state['winner'])
            self.assertFalse(state['optimization_accuracy_improved'])
            self.assertEqual(sum(c['kind']=='train' for c in calls),4)
            self.assertFalse(any(c['kind']=='audit' and 'test' in c['roles'] for c in calls))
            report=(root/'optimization/MIT/algorithm_comparison.md').read_text(encoding='utf-8')
            self.assertIn('unavailable (stage coverage)',report)
            self.assertIn('validation_a',report)

    def setup_fixture(self,root):
        plan={'datasets':['MIT'],'screen_seed':20260930,'confirmation_seeds':[20260931,20260932],
            'screen_candidates':[{'name':'bounded','override':{'local_recovery_max_soh':.005}},
                {'name':'query','override':{'local_readout':'query'}},{'name':'lora','override':{'numeric_lora_rank':4}},
                {'name':'huber','override':{'point_huber_beta_pp':.5}}],
            'score_weights':{'mae_pp':1.,'tail_mae_pp':1.,'stage_drop_max_pp':.5},
            'numeric_thresholds':{'mape_percent':5.,'mae_pp':2.,'tail_mae_pp':3.,'stage_drop_max_pp':3.},
            'screen_min_relative_improvement':.1,'confirmation_min_mean_relative_improvement':.1,
            'confirmation_min_improved_seed_count':2,'worst_val_mae_max_ratio':1.1}
        path=root/'plan.json';path.write_text(json.dumps(plan),encoding='utf-8');tasks=[]
        for fold in ('MIT_fold1','MIT_fold2'):
            for seed in range(20260930,20260935):
                settings=root/f'{fold}_{seed}.json';settings.write_text(json.dumps({'fold':fold,'seed':seed,
                    'local_readout':'vector','local_recovery_max_soh':.01,'point_huber_beta_pp':.1,'numeric_lora_rank':0}),encoding='utf-8')
                run=root/f'baseline_{fold}_{seed}';run.mkdir();self.metrics(run,10.)
                (run/'model.pt').write_bytes(b'CPU orchestration fixture; not model weights')
                tasks.append({'dataset':'MIT','fold':fold,'seed':seed,'settings':str(settings),'run':str(run),
                    'state':'complete','role_counts':{'train':4,'validation':2,'test':2}})
        return {'optimization_plan':str(path),'tasks':tasks}

    def metrics(self,run,score):
        # Point 2, tail 3, stage 6 -> score 8, for example.
        row={'evaluable':True,'mae_pp':score/4,'tail_mae_pp':score/4,'stage_drop_max_pp':score,
            'mape_percent':4. if score<10 else 6.}
        (run/'selected_metrics.json').write_text(json.dumps({'validation_a':row,'validation_b':row}),encoding='utf-8')

    def command_fixture(self,root,calls,all_fail=False):
        def command(queue_root,index,arguments,on_run=None):
            kind=arguments[1]
            if kind=='scripts.train_llm_curve_repair':
                cfg=json.loads(Path(arguments[arguments.index('--settings')+1]).read_text(encoding='utf-8'))
                score=11. if all_fail else (8. if cfg['local_recovery_max_soh']==.005 else 11. if cfg['numeric_lora_rank'] else 9.5)
                run=root/f'fixture_run_{len(calls)}';run.mkdir();self.metrics(run,score)
                (run/'model.pt').write_bytes(b'CPU orchestration fixture; not model weights')
                (run/'status.json').write_text(json.dumps({'state':'complete'}),encoding='utf-8')
                (run/'gradient_diagnostics.json').write_text(json.dumps([{'gradient_norms':{'backbone.attention.lora_B':.1}}]),encoding='utf-8')
                calls.append({'kind':'train','fold':cfg['fold'],'seed':cfg['seed']})
                if on_run:on_run(str(run))
                log=root/f'fixture_log_{len(calls)}.log';log.write_text('RUN='+str(run)+'\n',encoding='utf-8');return log
            roles=arguments[arguments.index('--roles')+1:]
            state=json.loads((root/'optimization/MIT/state.json').read_text(encoding='utf-8'))
            if 'test' in roles:
                self.assertEqual(state['state'],'formal_replication')
                self.assertEqual(state['winner'],'bounded')
                self.assertIn('locked_validation_evidence',state)
                lock=json.loads((root/'optimization/MIT/selection_lock.json').read_text(encoding='utf-8'))
                self.assertFalse(lock['TEST_requested_before_lock']);self.assertEqual(len(lock['candidate_evidence']),3)
            calls.append({'kind':'audit','roles':roles});return root/'unused.log'
        return command

    def test_screen_and_confirmation_never_request_test_then_reuse_three_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);queue=self.setup_fixture(root);calls=[]
            run_optimization(root,queue,'MIT',self.command_fixture(root,calls),Path('p0'),Path('p1'),Path('auth'))
            state=json.loads((root/'optimization/MIT/state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['state'],'complete');self.assertEqual(state['winner'],'bounded')
            self.assertEqual(len(state['formal_tasks']),10)
            self.assertEqual(sum(c['kind']=='train' for c in calls),4+2+7)
            self.assertEqual(sum(c['kind']=='audit' and 'test' in c['roles'] for c in calls),10)
            self.assertTrue(all(t['audited_roles']==['training','validation','test'] for t in state['formal_tasks']))
            before=len(calls)
            run_optimization(root,queue,'MIT',self.command_fixture(root,calls),Path('p0'),Path('p1'),Path('auth'))
            self.assertEqual(len(calls),before)

    def test_no_validation_winner_keeps_baseline_without_test_or_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);queue=self.setup_fixture(root);calls=[]
            run_optimization(root,queue,'MIT',self.command_fixture(root,calls,all_fail=True),Path('p0'),Path('p1'),Path('auth'))
            state=json.loads((root/'optimization/MIT/state.json').read_text(encoding='utf-8'))
            self.assertIsNone(state['winner']);self.assertFalse(state['optimization_accuracy_improved'])
            self.assertEqual(sum(c['kind']=='train' for c in calls),4)
            self.assertFalse(any(c['kind']=='audit' and 'test' in c['roles'] for c in calls))
            self.assertTrue((root/'optimization/MIT/algorithm_comparison.md').exists())

    def test_candidate_test_artifact_before_choice_is_locked_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);queue=self.setup_fixture(root);calls=[];base=self.command_fixture(root,calls)
            def premature_test(queue_root,index,arguments,on_run=None):
                log=base(queue_root,index,arguments,on_run)
                if arguments[1]=='scripts.train_llm_curve_repair':
                    run=Path(next(line[4:] for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('RUN=')))
                    (run/'test_audited_predictions.npz').write_bytes(b'forbidden premature fixture artifact')
                return log
            with self.assertRaisesRegex(ValueError,'TEST exists before selection lock'):
                run_optimization(root,queue,'MIT',premature_test,Path('p0'),Path('p1'),Path('auth'))
            self.assertFalse(any(c['kind']=='audit' and 'test' in c['roles'] for c in calls))


if __name__=='__main__':unittest.main()
