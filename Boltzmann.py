

def min_max_scl(v):
  return (v-v.min())/(v.max()-v.min())
def scale(v):
  return (v- v.mean())/v.std()


class BoltzmannModel(IsingModel):
  def __init__(self,n,p,beta, nb_cycle,prob,input_size, n_hidden_unit, eps = 1e-3):
    super().__init__(n,p,beta,nb_cycle,prob)
    self._input_size = input_size
    self._n_hidden_unit = n_hidden_unit
    self._W = np.random.random([n_hidden_unit,input_size])
    self._eps = eps
    self._c = np.random.random(n_hidden_unit)
    self._b = np.random.random(input_size)
    

  def  get_params(self):
    
    return {'n' : self._n,
            'p' : self._p,
            'proba' : self._proba,
            'beta' : self.beta,
            'nb_cycle' : self.nb_cycle,
            'input_size' : self._input_size,
            'n_hidden_unit' : self._n_hidden_unit,
            'eps' : self._eps}

  def __simul(self, v):
    cycle = 1
    mat  = ((2 * np.random.binomial(1,self._proba,len(x))-1)
    .reshape([self._n, self._p]))
    mat2 = mat.copy()

    while(cycle <= self.nb_cycle):
      cycle+=1
      #print(cycle)
      #walk_order = random.sample(list(range(self._n*self._p)),self._n*self._p)
      for i in range(self._n):
        for j in range(self._p):
          mat2 = self.__update(i,j,mat2)    
    return mat2

  def _forward(self,x):
    x = x.flatten()
    #h = np.random.random(self._n_hidden_unit)
    activation  = scale(self._c + self._W.dot(x))
    p_h_v = 1/(1+np.exp(-activation))
    h = np.array([np.random.binomial(1,p_h_v[i],1)[0] for i in range(len(p_h_v))])
    return h[:, None].T
  
  def _backward(self, h):
    activation= scale(self._b + h.dot(self._W))
    p_v_h = 1/(1+np.exp(-activation))
    v = np.array([np.random.binomial(1,p_v_h[i],1)[0] for i in range(len(p_v_h))])
    return v[:, None].T

  def get_w(self):
    return(self._W)

  def _CD_k(self, data, k = 10):
    h = np.zeros([len(data), self._n_hidden_unit])
    DW = np.zeros([self._n_hidden_unit, self._input_size])
    Db = np.zeros(self._input_size)
    Dc = np.zeros(self._n_hidden_unit)
    v = np.zeros( [len(data), self._input_size])

    v[0,] = min_max_scl(data[0].flatten())
    for t in range(len(data)-1):
      v_scl = min_max_scl(data[t].flatten())
      h[t,] = self._forward(v_scl)
      v[t+1,] = self._backward(h[t,])
    for i in range(self._n_hidden_unit):
      for j in range(self._input_size):
          DW[i,j] +=  (1/(1+np.exp(-(self._c + self._W.dot(v[0,])))))[0]*v[0,j] - (1/(1+np.exp(-(self._c + self._W.dot(v[k-1,])))))[0]*v[k-1,j]
          Db[j]+= v[0,j] - v[k-1,j]
          Dc[i]+= (1/(1+np.exp(-(self._c + self._W.dot(v[0,])))))[0] - (1/(1+np.exp(-(self._c + self._W.dot(v[k-1,])))))[0]
    self._W+=DW[0], 
    print(np.linalg.norm(DW))
    self._b+=Db
    self._c+=Dc

  def _update_w(self, v,h,v2,h2):
    W = np.array([list(i*h) for i in v])
    W2 = np.array([list(i*h2) for i in v2])
    DW = self._eps * (W-W2).T
    self._W+=DW
    self._b += v-v2
    self._c = self._W.dot(v)-self._W.dot(v2)


  def forward(self,v):
    v = v.flatten()
    v = (v-v.mean())/v.std()
    for cycle in tqdm(range(self.nb_cycle), desc ="epochs"):
      h = self._forward(v)
      #print(np.linalg.norm(self._W))
      v_prime = self._backward(h)
      h_prime = self._forward(v_prime)
      self._update_w(v,h,v_prime,h_prime)
    return self._W

  def CD_1(self,v):
    v = v[:,None].T

    mu = 1*(np.array([self._forward(v) for _ in range(100)]).mean(axis = 0) >  0.5)
    print(mu)
    h = self._forward(v)
    v_p = self._backward(h)
    print(v*h)
    mu_p = 1*(np.array([self._forward(v_p) for _ in range(100)]).mean(axis = 0) >  0.5)
    Dw = v.dot(mu)-v_p.dot(mu_p)
    #return Dw


  
  def forward2(self,v):
    v = min_max_scl(v)
    for cycle in tqdm(range(self.nb_cycle), desc ="epochs"):
      wi = self._W
      self._CD_k(v, k = 20)
    return self._W

def min_max_scl(v):
  return (v-v.min())/(v.max()-v.min())
def scale(v):
  return (v- v.mean())/v.std()


class BoltzmannModel(IsingModel):
  def __init__(self,n,p,beta, nb_cycle,prob,input_size, n_hidden_unit, eps = 1e-3):
    super().__init__(n,p,beta,nb_cycle,prob)
    self._input_size = input_size
    self._n_hidden_unit = n_hidden_unit
    self._W = np.random.random([n_hidden_unit,input_size])
    self._eps = eps
    self._c = np.random.random(n_hidden_unit)
    self._b = np.random.random(input_size)
    

  def  get_params(self):
    
    return {'n' : self._n,
            'p' : self._p,
            'proba' : self._proba,
            'beta' : self.beta,
            'nb_cycle' : self.nb_cycle,
            'input_size' : self._input_size,
            'n_hidden_unit' : self._n_hidden_unit,
            'eps' : self._eps}

  def __simul(self, v):
    cycle = 1
    mat  = ((2 * np.random.binomial(1,self._proba,len(x))-1)
    .reshape([self._n, self._p]))
    mat2 = mat.copy()

    while(cycle <= self.nb_cycle):
      cycle+=1
      #print(cycle)
      #walk_order = random.sample(list(range(self._n*self._p)),self._n*self._p)
      for i in range(self._n):
        for j in range(self._p):
          mat2 = self.__update(i,j,mat2)    
    return mat2

  def _forward(self,x):
    x = x.flatten()
    #h = np.random.random(self._n_hidden_unit)
    activation  = scale(self._c + self._W.dot(x))
    p_h_v = 1/(1+np.exp(-activation))
    h = np.array([np.random.binomial(1,p_h_v[i],1)[0] for i in range(len(p_h_v))])
    return h[:, None].T
  
  def _backward(self, h):
    activation= scale(self._b + h.dot(self._W))
    p_v_h = 1/(1+np.exp(-activation))
    v = np.array([np.random.binomial(1,p_v_h[i],1)[0] for i in range(len(p_v_h))])
    return v[:, None].T

  def get_w(self):
    return(self._W)

  def _CD_k(self, data, k = 10):
    h = np.zeros([len(data), self._n_hidden_unit])
    DW = np.zeros([self._n_hidden_unit, self._input_size])
    Db = np.zeros(self._input_size)
    Dc = np.zeros(self._n_hidden_unit)
    v = np.zeros( [len(data), self._input_size])

    v[0,] = min_max_scl(data[0].flatten())
    for t in range(len(data)-1):
      v_scl = min_max_scl(data[t].flatten())
      h[t,] = self._forward(v_scl)
      v[t+1,] = self._backward(h[t,])
    for i in range(self._n_hidden_unit):
      for j in range(self._input_size):
          DW[i,j] +=  (1/(1+np.exp(-(self._c + self._W.dot(v[0,])))))[0]*v[0,j] - (1/(1+np.exp(-(self._c + self._W.dot(v[k-1,])))))[0]*v[k-1,j]
          Db[j]+= v[0,j] - v[k-1,j]
          Dc[i]+= (1/(1+np.exp(-(self._c + self._W.dot(v[0,])))))[0] - (1/(1+np.exp(-(self._c + self._W.dot(v[k-1,])))))[0]
    self._W+=DW[0], 
    print(np.linalg.norm(DW))
    self._b+=Db
    self._c+=Dc

  def _update_w(self, v,h,v2,h2):
    W = np.array([list(i*h) for i in v])
    W2 = np.array([list(i*h2) for i in v2])
    DW = self._eps * (W-W2).T
    self._W+=DW
    self._b += v-v2
    self._c = self._W.dot(v)-self._W.dot(v2)


  def forward(self,v):
    v = v.flatten()
    v = (v-v.mean())/v.std()
    for cycle in tqdm(range(self.nb_cycle), desc ="epochs"):
      h = self._forward(v)
      #print(np.linalg.norm(self._W))
      v_prime = self._backward(h)
      h_prime = self._forward(v_prime)
      self._update_w(v,h,v_prime,h_prime)
    return self._W

  def CD_1(self,v):
    v = v[:,None].T

    mu = 1*(np.array([self._forward(v) for _ in range(100)]).mean(axis = 0) >  0.5)
    print(mu)
    h = self._forward(v)
    v_p = self._backward(h)
    print(v*h)
    mu_p = 1*(np.array([self._forward(v_p) for _ in range(100)]).mean(axis = 0) >  0.5)
    Dw = v.dot(mu)-v_p.dot(mu_p)
    #return Dw


  
  def forward2(self,v):
    v = min_max_scl(v)
    for cycle in tqdm(range(self.nb_cycle), desc ="epochs"):
      wi = self._W
      self._CD_k(v, k = 20)
    return self._W