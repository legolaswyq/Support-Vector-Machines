import numpy as np

a = [1,4,6,2,3]
index = [b[0] for b in sorted(enumerate(a),key=lambda i:i[1])]
print(index)