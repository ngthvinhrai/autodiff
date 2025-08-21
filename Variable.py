import math
import random
import numpy as np

class variable:
    def __init__(self, data=None, parents=[], require_grad=True):
        self.data = data if data else random.random()
        self.parents = parents
        self.childs = []
        self.require_grad = require_grad
        self.grad = 0
        self.bw = lambda: None
        
    def __add__(self, other):
        other = other if isinstance(other, variable) else variable(other, require_grad=False)
        out = variable(self.data + other.data, parents=[self, other])
        self.childs.append(out)
        other.childs.append(out)
        
        def _backward():
            if self.require_grad: self.grad += out.grad    
            if other.require_grad: other.grad += out.grad
        out.bw = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, variable) else variable(other, require_grad=False)
        out = variable(self.data - other.data, parents=[self, other])
        self.childs.append(out)
        other.childs.append(out)

        def _backward():
            if self.require_grad: self.grad += out.grad  
            if other.require_grad: other.grad += -out.grad
        out.bw = _backward

        return out
    
    def __rsub__(self, other):
        return variable(-self.data + other)

    def __mul__(self, other):
        other = other if isinstance(other, variable) else variable(other, require_grad=False)
        out = variable(self.data * other.data, parents=[self, other])
        self.childs.append(out)
        other.childs.append(out)

        def _backward():
            if self.require_grad: self.grad += out.grad * other.data  
            if other.require_grad: other.grad += out.grad * self.data
        out.bw = _backward

        return out
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = other if isinstance(other, variable) else variable(other, require_grad=False)
        out = variable(self.data ** other.data, parents=[self, other])

        def _backward():
            if self.require_grad: self.grad += out.grad * other.data * self.data**(other.data - 1)
            if other.require_grad: other.grad += out.grad * out.data * math.log(self.data)
        out.bw = _backward

        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, variable) else variable(other, require_grad=False)
        return self*(other**-1)
    
    def __rtruediv__(self, other):
        other = other if isinstance(other, variable) else variable(other, require_grad=False)
        return other*(self**-1)

    def __neg__(self):
        return variable(-self.data)

    def exp(self):
        out = variable(math.exp(self.data), parents=[self])
        self.childs.append(out)

        def _backward():
            if self.require_grad: self.grad += out.grad * out.data
        out.bw = _backward

        return out

    def log(self):
        out = variable(math.log(self.data), parents=[self])
        self.childs.append(out)
        
        def _backward():
            if self.require_grad: self.grad += out.grad * 1/self.data
        out.bw = _backward

        return out
    
    def sin(self):
        out = variable(math.sin(self.data), parents=[self])
        self.childs.append(out)

        def _backward():
            if self.require_grad: self.grad += out.grad * math.cos(self.data)
        out.bw = _backward

        return out

    def cos(self):
        out = variable(math.cos(self.data), parents=[self])
        self.childs.append(out)
        
        def _backward():
            if self.require_grad: self.grad += out.grad * -math.sin(self.data)
        out.bw = _backward

        return out

    def relu(self):
        value = self.data if self.data > 0 else 0
        out = variable(value, parents=[self])
        self.childs.append(out)

    def topo_traverse(self, visited):
        topo = []

        if self not in visited:
            visited.add(self)
            for par in self.parents:
                topo.extend(par.topo_traverse(visited))
            topo.append(self)

        return topo

    def backward(self):
        self.grad = 1
        topo = self.topo_traverse(visited=set())

        for v in reversed(topo): v.bw()

    def __repr__(self):
        return f"variable(={self.data})"
    
class vector:
    def __init__(self, data=None, dim=None):
        if data:
            self.data = np.array([variable(x) for x in data])
            self.shape = self.data.shape
        else:
            self.data = np.array([variable() for _ in range(dim)])
            self.shape = self.data.shape


    def __add__(self, other):
        assert self.shape == other.shape, f"Not same shape: {self.shape[0]} != {other.shape[0]}" 

        out = vector(dim=self.shape[0])
        out.data = self.data + other.data

        return out

    def __repr__(self):
        return f"variable_vector(={[d.data for d in self.data]})"
    

class matrix:
    def __init__(self):
        pass
    
def softmax():
    x1 = variable(5)
    x2 = variable(2)
    l1 = x1.exp()
    l2 = x2.exp()
    denominator = l1 + l2
    s1 = l1/denominator
    s2 = l2/denominator
    s = s1 * s2
    s.backward()
    print()

def main():
    x = vector([1,2,3])
    y = vector([4,5,6])
    print(x+y)

if __name__ == "__main__":
    main()

    