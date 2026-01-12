ckpt_path=${1:-"/m-coriander/coriander/banghua/shirui/d1/diffu_grpo/checkpoints/math500_p_iter1_beta0.0_lladou_onpolicy_peftfalse_lr5e-6_block256_len256_advclip0.0_temp.1_freezeunmaskheadfalse/checkpoint-15000"}

repo_id="sengi/lladou-math500"

python scripts/upload_checkpoint.py \
        --checkpoint-dir $ckpt_path \
        --repo-id $repo_id \
        --branch main
