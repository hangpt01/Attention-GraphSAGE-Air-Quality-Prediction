# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1 --num_epochs_decoder 1 --dataset "beijing"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 0 --num_epochs_decoder 10 --dataset "beijing" 
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 0 --num_epochs_decoder 50 --dataset "beijing" 
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 50 --num_epochs_decoder 50 --dataset "beijing"  --lr_stdgi 0.0005 #all feats
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1 --num_epochs_decoder 50 --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 300 --num_epochs_decoder 300 --lr_stdgi 0.00001 --lr_decoder 0.000005 --dataset "uk"
# MAE 1.81 with PM2.5, windspeed
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0005 --lr_decoder 0.00001 --features "PM2.5,O3,NO2,SO2,PM10,2m_temperature,surface_pressure,evaporation,total_precipitation,wind_speed" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.00005 --lr_decoder 0.000005 --features "PM2.5,O3,NO2,SO2,PM10,2m_temperature,surface_pressure,evaporation,total_precipitation,wind_speed" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 0 --num_epochs_decoder 1000 --lr_stdgi 0.00005 --lr_decoder 0.000001 --features "PM2.5,O3,NO2,SO2,PM10,2m_temperature,surface_pressure,evaporation,total_precipitation,wind_speed" --dataset "uk"

#MAE 1.81 with PM2.5,windspeed
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0001 --lr_decoder 0.00001 --features "PM2.5,wind_speed" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0001 --lr_decoder 0.00001 --features "PM2.5,O3" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0001 --lr_decoder 0.00001 --features "PM2.5,evaporation" --dataset "uk"
CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0005 --lr_decoder 0.00001 --features "PM2.5,NO2" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0001 --lr_decoder 0.00001 --features "PM2.5,PM10" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0001 --lr_decoder 0.00001 --features "PM2.5,total_precipitation" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0001 --lr_decoder 0.00001 --features "PM2.5" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0001 --lr_decoder 0.00001 --features "PM2.5,2m_temperature" --dataset "uk"
# CUDA_VISIBLE_DEVICES=0 python main.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 1000 --num_epochs_decoder 1000 --lr_stdgi 0.0001 --lr_decoder 0.00001 --features "PM2.5,SO2" --dataset "uk"




