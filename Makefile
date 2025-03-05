IMAGE_NAME = ft_linear_regression

build:
	docker build -t $(IMAGE_NAME) .

train:
	@docker run --rm -it -v $(PWD)/app:/app -w /app -e PYTHONPATH=/app $(IMAGE_NAME) python train.py

predict:
	@docker run --rm -it -v $(PWD)/app:/app -w /app -e PYTHONPATH=/app $(IMAGE_NAME) python predict.py

evaluate:
	@docker run --rm -it -v $(PWD)/app:/app -w /app -e PYTHONPATH=/app $(IMAGE_NAME) python evaluate.py

test:
	@docker run --rm -it -v $(PWD)/app:/app -w /app -e PYTHONPATH=/app $(IMAGE_NAME) pytest --tb=short -v -s


clean:
	docker system prune -f

purge:
	docker rmi $(IMAGE_NAME)
