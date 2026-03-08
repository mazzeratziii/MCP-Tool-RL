import sys
print(f'Python version: {sys.version}')

try:
    import numpy
    print(f'✓ numpy {numpy.__version__}')
except Exception as e:
    print(f'✗ numpy: {e}')

try:
    import torch
    print(f'✓ torch {torch.__version__}')
except Exception as e:
    print(f'✗ torch: {e}')

try:
    from datasets import load_dataset
    print(f'✓ datasets')
except Exception as e:
    print(f'✗ datasets: {e}')

try:
    from sentence_transformers import SentenceTransformer
    print(f'✓ sentence-transformers')
except Exception as e:
    print(f'✗ sentence-transformers: {e}')

try:
    from toolbrain import Brain
    print(f'✓ toolbrain')
except Exception as e:
    print(f'✗ toolbrain: {e}')