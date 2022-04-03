
class BoltzmannModel(IsingModel):
  def __init__(self,n,p,beta, nb_cycle,prob,input_size, n_hidden_unit, eps = 1e-3):
    super().__init__(n,p,beta,nb_cycle,prob)
    self._input_size = input_size
    self._n_hidden_unit = n_hidden_unit
    self._W = np.random.random([n_hidden_unit,input_size])
    self._eps = eps 

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
    h = np.random.random(self._n_hidden_unit)
    #pX_ij_cond_Nij  = 1/(1+np.exp(-2*self.beta * sum_N_index))
    bias = np.random.random(self._n_hidden_unit)
    #print(f'bias : {bias}')
    #print(f'w*x : {self._W.dot(x)}')
    #print(self._W)
    p_h_v = 1/(1+np.exp(-(bias + self._W.dot(x))))
    #print(p_h_v)
    h = np.array([np.random.binomial(1,p_h_v[i],1)[0] for i in range(len(p_h_v))])
    return h
  
  def _update_w(self, v,h,v2,h2):
    W = np.array([list(i*h) for i in v])
    W2 = np.array([list(i*h2) for i in v2])
    DW = self._eps * (W-W2).T
    self._W+=DW

  def _backward(self, h):
    bias =np.random.random(self._input_size)
    p_v_h = 1/(1+np.exp(-(bias + h.T.dot(self._W))))
    v = np.array([np.random.binomial(1,p_v_h[i],1)[0] for i in range(len(p_v_h))])
    return v

  def get_w(self):
    return(self._W)

  def forward(self,v):
    v = v.flatten()
    v = (v-v.mean())/v.std()
    for cycle in tqdm(range(self.nb_cycle), desc ="epochs"):
      h = self._forward(v)
      #print(f'h = {h}')
      print(np.linalg.norm(self._W))
      v_prime = self._backward(h)
      h_prime = self._forward(v_prime)
      self._update_w(v,h,v_prime,h_prime)
    return self._W