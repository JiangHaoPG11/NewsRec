debug_flag=true
model_name=RCENR

if [ "$debug_flag" == true ]; then
    python main.py \
        --mode $model_name 
else
    nohup python main.py \
        --mode $model_name  > ${model_name}_result.log &
fi