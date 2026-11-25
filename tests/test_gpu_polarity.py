"""Cached-model bounded polarity setup and actual optimizer-step fixture."""
import os
import pytest
import torch
pytestmark=pytest.mark.skipif(os.environ.get('RUN_GPU')!='1',reason='set RUN_GPU=1')


def test_pinned_setup_and_positive_update():
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model,LoraConfig
    from rl.campaign import MODEL_REVISION,assert_model_setup
    import train_grpo as tg
    model=get_peft_model(AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',
                           revision=MODEL_REVISION,local_files_only=True,dtype=torch.bfloat16).cuda(),
                         LoraConfig(r=16,lora_alpha=32,target_modules='all-linear',lora_dropout=0.0))
    audit=assert_model_setup(model)
    print(audit['trainable_parameters'],len(audit['targets']),audit['lora_dtype'],audit['base_dtype'])
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.99),weight_decay=0)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/10))
    args=tg.parse_args(['--run','fixture','--advantage-mode','positive','--diagnostic-updates','1'])
    args.current_update=1
    groups=[{'idx':0,'seqs':[{'prompt_ids':torch.tensor([100,101]),'comp_ids':torch.tensor([102,103,151645]),
                            'length':3,'reward':float(i<3)} for i in range(8)]}]
    result=tg.train_step(model,opt,sched,groups,args,'cuda',151643)
    assert result['parameter_rms_change']>0 and result['active_positive_answers']==3
    assert result['active_negative_answers']==0 and result['trained_answers']==8
    assert result['lr']==pytest.approx(1e-5)
    assert all(not p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n)
    print({k:result[k] for k in ['lr','parameter_rms_change','grad_norm','policy_grad_norm','weighted_kl_grad_norm']})
