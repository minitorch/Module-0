import os, sys
if len(sys.argv) == 1 or int(sys.argv[1]) == 0:
	print('')
	print('=================')
	print('Running test 0')
	print('=================')
	os.system('flake8 --ignore "N801, E203, E266, E501, W503, F812, E741, N803, N802, N806" minitorch/ tests/ project/')
if len(sys.argv) == 1 or int(sys.argv[1]) == 1:
	print('')
	print('=================')
	print('Running test 1')
	print('=================')
	os.system('python -m pytest tests/ -m task0_1')
if len(sys.argv) == 1 or int(sys.argv[1]) == 2:
	print('')
	print('=================')
	print('Running test 2')
	print('=================')
	os.system('python -m pytest tests/ -m task0_2')
if len(sys.argv) == 1 or int(sys.argv[1]) == 3:
	print('')
	print('=================')
	print('Running test 3')
	print('=================')
	os.system('python -m pytest tests/ -m task0_3')
if len(sys.argv) == 1 or int(sys.argv[1]) == 4:
	print('')
	print('=================')
	print('Running test 4')
	print('=================')
	os.system('python -m pytest tests/ -m task0_4')
