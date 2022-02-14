A very simple Neural Network console app written from scatch in c#.

Supervised learning, Gradient decent and Sigmoid activations.

This project is a testing enviroment to learn more about Machine Learning. It is very simple and has inbuilt data sets for solving of some simple problems:

Inbuilt test data sets in Testdata.cs: "alpha", "xor", "xnor"
Selected dataset can be changed here:
```
Program.cs
13  var testData = new TestData("alpha");
```

Network Topology can be changed with this line. The input and output layers are automaticaly set depending on test data selected.
This example is a 3 layerd network with 10 hidden nodes on 1 hidden layer.
The commented line would be a network with 5 layers, 3 hidden layers with 20 nodes on each.
```
Program.cs
14  var net = new NeuralNet(new int[] { testData.inputs[0].Length, 10, testData.targets[0].Length });
15  //var net = new NeuralNet(new int[] { testData.inputs[0].Length, 20, 20, 20, testData.targets[0].Length });
```

The learning rate is currently hardcoded here
```
Layer.cs
37 public void UpdateWeights()
       ....
43     weights[i, j] -= weightsDelta[i, j] * 0.1f;
       ...
```
