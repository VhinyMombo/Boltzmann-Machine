n = 50
p =  50
prob = 0.5
beta = 10
nb_cycle = 100
model = IsingModel(n,p,beta, nb_cycle,prob)
plt.imshow(model())