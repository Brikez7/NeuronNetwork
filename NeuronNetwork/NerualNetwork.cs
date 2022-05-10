namespace NeuralNetworks
{
    public class NerualNetwork
    {
        public List<Layer> Layers { get; } = new List<Layer>();
        private TopologeNetwork Topologe{ get; }

        public NerualNetwork(TopologeNetwork topologe)
        {
            Topologe = topologe;
            CreateInputLayers();
            CreateHiddenLayers();
            CreateOutputLayers();
        }

        private void CreateInputLayers(int CountInputsNeuron = 1) 
        {
            List<Neuron> inputsNeurons = new List<Neuron>();
            for (int i = 0; i < Topologe.CountInputs; i++)
            {
                Neuron neuron = new Neuron(CountInputsNeuron, NeuronType.input);
                inputsNeurons.Add(neuron);
            }
            Layer inputLayer = new Layer(inputsNeurons.ToArray(),NeuronType.input);
            Layers.Add(inputLayer);
        }
        private void CreateOutputLayers()
        {
            List<Neuron> outputsNeurons = new List<Neuron>();
            Layer lastLayer = Layers.Last();
            for (int i = 0; i < Topologe.CountOutputs; i++)
            {
                Neuron neuron = new Neuron(lastLayer.CountNeurons, NeuronType.output);
                outputsNeurons.Add(neuron);
            }
            Layer outputLayer = new Layer(outputsNeurons.ToArray(), NeuronType.output);
            Layers.Add(outputLayer);
        }
        private void CreateHiddenLayers()
        {
            for (int i = 0; i < Topologe.HiddingLayers.Length; i++)
            {
                List<Neuron> hiddenNeurons = new List<Neuron>();
                Layer lastLayer = Layers.Last();
                for (int ii = 0; ii < Topologe.HiddingLayers[i]; ii++)
                {
                    Neuron neuron = new Neuron(lastLayer.CountNeurons);
                    hiddenNeurons.Add(neuron);
                }
                Layer hiddenLayer = new Layer(hiddenNeurons.ToArray(), NeuronType.Normal);
                Layers.Add(hiddenLayer);
            }
        }

        public Neuron FeedForward(params double[] inputsSignals)
        {
            SendSignalsOnInputsNeurons(inputsSignals);
            ProcessingSignals();
            if (Topologe.CountOutputs > 1)
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
            else 
            {
                return Layers.Last().Neurons[0];
            }
        }

        private void ProcessingSignals()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                Layer layer = Layers[i];
                double[] signals = Layers[i - 1].GetSignals();
                foreach (var neuron in layer.Neurons)
                {
                    neuron.ProcessingSignal(signals);
                }
            }
        }

        private void SendSignalsOnInputsNeurons(params double[] InputsSignalsAtNeuron)
        {
            for (int i = 0; i < InputsSignalsAtNeuron.Length; i++)
            {
                Neuron neuron = Layers[0].Neurons[i];

                neuron.ProcessingSignal(InputsSignalsAtNeuron[i]);
            }
        }
        private double BackPropagetion(double exprected, params double[] inputs) 
        {
            double result = FeedForward(inputs).Output;
            double difference = result - exprected;
            foreach (Neuron neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference,Topologe.LearningRate);
            }

            for (int i = Layers.Count - 2; i > 0; i--)
            {
                Layer currentLayer = Layers[i];
                Layer previousLayer = Layers[i + 1];


                for (int x = 0; x < currentLayer.CountNeurons; x++)
                {
                    Neuron currentNeuron = currentLayer.Neurons[x];

                    for (int y = 0; y < previousLayer.CountNeurons; y++)
                    {
                        Neuron previousNeuron = previousLayer.Neurons[y];
                        double error = previousNeuron.Weight[x] * currentNeuron.Delta;
                        previousNeuron.Learn(error, Topologe.LearningRate);
                    }
                }
            }
            return Math.Pow(difference,2);
        }
        // здесь что то странное
        public double Learn(List<Tuple<double, double[]>> dataSet,double period) 
        {
            double error = 0.0;
            for (int i = 0; i < period; i++)
            {
                foreach (var data in dataSet)
                {
                    error += BackPropagetion(data.Item1,data.Item2);
                }
            }

            return error / period;
        }
        static async Task Main() 
        {
            Console.WriteLine();
        }
    }
}
