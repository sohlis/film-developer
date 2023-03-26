.PHONY: install run

install:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "Installing dependencies..."
	venv/bin/pip install -r requirements.txt

run:
	@echo "Running film_emulation.py script..."
	venv/bin/python3 film_emulation.py input_images output_images
