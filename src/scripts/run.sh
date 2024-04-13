debug_flag=true
model_name=MCCM
epoch=10
data_scale=2wU

if [ "$debug_flag" == true ]; then
    python main.py \
        --data_scale $data_scale \
        --mode $model_name \
        --epoch $epoch \
        --train_flag true \
        --test_flag true 
else
    nohup python main.py \
        --data_scale $data_scale \
        --mode $model_name \
        --epoch $epoch \
        --train_flag true \
        --test_flag true   > ${model_name}_${data_scale}_epoch_${epoch}_result_v1.log &
fi