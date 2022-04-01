
class BoltzmannModel(IsingModel):
  def __init__(self,n,p,beta, nb_cycle,prob,input_size, n_hidden_unit):
    super().__init__(n,p,beta,nb_cycle,prob)
    self._input_size = input_size
    self._n_hidden_unit = n_hidden_unit

  def  get_params(self):
    
    return {'n' : self._n,
            'p' : self._p,
            'proba' : self._proba,
            'beta' : self.beta,
            'nb_cycle' : self.nb_cycle,
            'input_size' : self._input_size,
            'n_hidden_unit' : self._n_hidden_unit}

    def __simul(self, v):
      cycle = 1
      mat  = ((2 * np.random.binomial(1,self._proba,self._n * self._p)-1)
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

  def forward(self,x):
    x = x.flatten()
    h = np.random.random(self._n_hidden_unit)





  