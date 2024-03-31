debug_flag=false
model_name=RCENR
epoch=60

if [ "$debug_flag" == true ]; then
    python main.py \
        --mode $model_name \
        --epoch $epoch \
        --train_flag true \
        --test_flag true 
else
    nohup python main.py \
        --mode $model_name \
        --epoch $epoch \
        --train_flag true \
        --test_flag true   > ${model_name}_epoch_${epoch}_result.log &
fi