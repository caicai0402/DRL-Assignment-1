run:
	python simple_custom_taxi_env.py \
		--grid_size 10 \
		--fuel_limit 5000 \
		--obstacles_percentage 0.2

train:
	python train_qtable_agent.py \
		--grid_size 10 \
		--fuel_limit 5000 \
		--obstacles_percentage 0 \
		--pretrained_model results/qtable.pkl \
		--save_path results/qtable.pkl

train_pg:
	python train_policy_table_agent.py \
		--grid_size 10 \
		--fuel_limit 5000 \
		--obstacles_percentage 0 \
		--pretrained_model results/policy_table.pkl \
		--save_path results/policy_table.pkl

temp:
	python train_lstmppo_agent.py \
		--grid_size 10 \
		--fuel_limit 5000 \
		--obstacles_percentage 0.2 \
		--save_path results/lstmppo.pth

test:
	python train_qtable_agent_test.py \
		--grid_size 10 \
		--fuel_limit 5000 \
		--obstacles_percentage 0 \
		--pretrained_model qtable.pkl \
		--save_path qtable.pkl