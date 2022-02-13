using System;
using System.Collections;
using System.Diagnostics;
using System.Threading;

namespace SimpleNN
{
    class Program
    {
        static void Main(string[] args)
        {
            //Console.WriteLine("Hello World!");
            var testData = new TestData("alpha");
            var net = new NeuralNet(new int[] { testData.inputs[0].Length, 10, testData.targets[0].Length });

            Stopwatch watch = new Stopwatch();
            for (int i = 0; i < 10000000000; i++)
            {

                float[] targs;
                string label;
                float[] ins = testData.NextRandomDataSet(out targs, out label);
                watch.Restart();
                var res = net.FeedForwardNetwork(ins);
                net.BackPropNetwork(targs);
                watch.Stop();
                
                if (i % 1000 == 0)
                {
                    float error = 0;
                    Console.SetCursorPosition(0, 0);
                   // Console.Clear();
                    for (int j = 0; j < net.layers[^1].error.Length; j++)
                    {
                        error += Math.Abs(net.layers[^1].error[j]);
                    }

                    Console.Write("Inputs: ");
                    foreach (var item in ins)
                    {
                        Console.Write("," + item);
                    }
                    Console.Write("  Label: " + label + "\n");
                    //Console.Write("\n");

                    Console.Write("Outputs: ");
                    foreach (var item in res)
                    {
                        Console.Write("," + item);
                    }
                    Console.Write("\n");

                    Console.Write("Targets: ");
                    foreach (var item in targs)
                    {
                        Console.Write("," + item);
                    }
                    Console.Write("\n");
                    Console.Write("Guess: ");
                    for (int j = 0; j < res.Length; j++)
                    {
                        if (res[j] > 0.9)
                        {
                            Console.Write(testData.outputLabels[j] + " ");
                        }
                    }



                    Console.WriteLine("Error: " + error);
                    Console.WriteLine("Time: " + watch.Elapsed.TotalMilliseconds);

                    /*float[] res;
                    Console.SetCursorPosition(0, 0);
                    res = net.FeedForwardNetwork(new float[] { 0, 0, 1 });
                    Console.WriteLine("In 0,0  Target: 1,0  Actual: " + res[0] + "," + res[1]);
                    res = net.FeedForwardNetwork(new float[] { 0, 1, 1 });
                    Console.WriteLine("In 0,1  Target: 0,1  Actual: " + res[0] + "," + res[1]);
                    res = net.FeedForwardNetwork(new float[] { 1, 0, 1 });
                    Console.WriteLine("In 1,0  Target: 0,1  Actual: " + res[0] + "," + res[1]);
                    res = net.FeedForwardNetwork(new float[] { 1, 1, 1 });
                    Console.WriteLine("In 1,1  Target: 1,0  Actual: " + res[0] + "," + res[1]);
                    */

                    Console.WriteLine("Run: " + i);
                    //Thread.Sleep(200);
                }
                         
            }

            
            
            Console.WriteLine("Done");
        }
    }
}
