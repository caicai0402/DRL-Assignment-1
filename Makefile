run:
	python simple_custom_taxi_env.py \
		--grid_size 5 \
		--fuel_limit 5000 \
		--obstacles_percentage 0.1

train:
	python train_agent.py \
		--grid_size 5 \
		--fuel_limit 5000 \
		--obstacles_percentage 0.2 \
		--pretrained_model q_table.pkl \
		--save_path q_table.pkl

temp:
	python train_lstmppo_agent.py \
		--grid_size 5 \
		--fuel_limit 5000 \
		--obstacles_percentage 0 \
		--pretrained_model lstmppo.pth \
		--save_path lstmppo.pth
