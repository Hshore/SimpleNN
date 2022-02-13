using System;
using System.Collections.Generic;
using System.Text;

namespace SimpleNN
{
     
    class NeuralNet
    {

        int[] layer;
        public Layer[] layers;
        
        public NeuralNet(int[] layer)
        {

            this.layer = new int[layer.Length];
            for (int i = 0; i < layer.Length; i++)
            {
                this.layer[i] = layer[i];
            }

            layers = new Layer[layer.Length - 1];

            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(layer[i], layer[i + 1]);
            }

        }

        public float[] FeedForwardNetwork(float[] inputs)
        {
            layers[0].FeedForwardLayer(inputs);
            for (int i = 1; i < layers.Length; i++)
            {
                layers[i].FeedForwardLayer(layers[i - 1].outputs);
            }


            return layers[layers.Length - 1].outputs;
        }

        public void BackPropNetwork(float[] expected)
        {
            for (int i = layers.Length-1; i >= 0; i--)
            {
                if (i == layers.Length - 1)
                {
                    layers[i].BackPropOutput(expected);
                }
                else
                {
                    layers[i].BackPropHidden(layers[i + 1].gamma, layers[i + 1].weights);
                }

                layers[i].UpdateWeights();
            }

            //for (int i = 0; i < layers.Length; i++)
            //{
           // }
        } 

    }
}
