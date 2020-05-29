clean_sdist:
	rm -rf carla_birdeye_view.egg-info
	rm -rf dist

build_sdist: clean_sdist
	python setup.py sdist

upload: build_sdist
	twine upload --verbose dist/*
