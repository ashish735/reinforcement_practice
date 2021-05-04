import torch.nn as nn # it gives access to the layers
import torch.nn.functional as F #it gives access to function
import torch.optim as optim # it gives access to optimizers
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifer, self).__init__()

        # fc stands for fully connected layer
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # Using Adam optimizer - a type of SGD with momentum
        # self.parameters() tells us what we want to be optimizing
        # self.parameters() comes from nn.Module
        # it returns an iterator over Module Parameters.
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        #sending the varialbe tensors to the gpu or cpu
        self.to(self.device)

    # pytorch handles the back propagation algorithm for us
    # but we have to define forward propagation by ourself
    # function will take some data as input
    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))

        # here we are not using F because CrossEntropyLoss will handle
        # the activation function here
        layer3 = self.fc3(layer2)
        return layer3

    # function will take input data and corresponding labels
    def learn(self, data, labels):
        # we have to zero out the gradients for optimizer
        # because pytorch keeps track of the gradients between
        # learning loops, we dont want any cross chatter
        # between the gradients from one iteration of the learning loop
        # to the other
        self.optimizer.zero_grad()

        # convert the numpy data types into torch tensors data types
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)

        # there is also T.Tensor() it preserves the data types if it is float32

        predictions = self.forward(data)

        cost = self.loss(predictions, labels)

        #now we have to back propagate the cost
        #it calls the differentiation function
        cost.backward()

        # performs a single optimization step
        # i.e updates the parameter
        self.optimizer.step()


    """while(True):
        data_random=bigdata.sample(n=100)
        data_random=data_random.reset_index(drop=True)
        X=data_random.loc[:,0:12] 
        y=data_random.loc[:,13]

        ## self.optimizer.zero_grad()
        finalW=0.0    
        finalB=0.0


        for j in range(100):

            ## cost or loss = (y.loc[j]-np.dot(X.loc[j],w.T)-b)

            ## cost.backward()  i.e calculating gradients
            ## finalW+= -2*X.loc[j]*cost
            ## finalB+= float(-2*cost)
            ## shape of finalW and X.loc[j] is same
            
            finalW+=-2*X.loc[j]*(y.loc[j]-np.dot(X.loc[j],w.T)-b)
            finalB+=float(-2*(y.loc[j]-np.dot(X.loc[j],w.T)-b))  

        # finalW/100 = averaging the gradients
        # self.optimizer.step() i.e updating the parameters using calculated gradients
        # we are subtracting because
        ## Thus we can say that the gradient at any point of a function points in the 
        ## direction of steepest ascent, i.e. if I were to climb the function as quickly as possible, 
        ## I would choose the direction of the gradient.
        ## While in gradient descent algorithm, we wish to choose to move in the direction where the 
        ## function (i.e. the loss function) reduces the maximum, thus we move in the opposite direction 
        ## of the gradient, i.e. the direction of steepest descent. Therefore, we subtract the gradient in 
        ## the gradient descent algorithm.
        
        w0=w-r*(finalW/100)
        b0=b-r*(finalB/100)

        # decay of learning  rate
        r=r/2

        # checking for convergence
        if(np.array(w)==np.array(w0)).all():
            break;
        else:
            w=w0
            b=b0"""