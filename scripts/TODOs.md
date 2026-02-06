# TODOs - scripts

- [ ] "--cov-report=term-missing", - `scripts\run_tests.py:103`
- [ ] class MockModel: - `scripts\train_synthetic.py:78`
- [ ] Mock model for synthetic training validation. - `scripts\train_synthetic.py:80`
- [ ] Simulates forward pass and loss computation without actual model. - `scripts\train_synthetic.py:82`
- [ ] # Initialize mock weights - `scripts\train_synthetic.py:89`
- [ ] # Embedding lookup (simulated) - `scripts\train_synthetic.py:117`
- [ ] # Create mock model - `scripts\train_synthetic.py:257`
- [ ] model = MockModel(config, backend) - `scripts\train_synthetic.py:258`
- [ ] return False, f"Missing {description}: {path}" - `scripts\validate_project.py:28`
- [ ] results.append((cpu_path.exists(), f"Backend CPU: {'file exists' if cpu_path.exists() else 'missing'}")) - `scripts\validate_project.py:63`
- [ ] results.append((gpu_path.exists(), f"Backend GPU: {'file exists' if gpu_path.exists() else 'missing'}")) - `scripts\validate_project.py:66`
- [ ] results.append((tpu_path.exists(), f"Backend TPU: {'file exists' if tpu_path.exists() else 'missing'}")) - `scripts\validate_project.py:69`
- [ ] results.append((False, f"Missing tests directory: {subdir}/")) - `scripts\validate_project.py:111`
