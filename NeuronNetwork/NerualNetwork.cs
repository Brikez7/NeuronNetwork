namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        private TopologyNetwork Topology { get; }
        private List<Layer> Layers { get; } = new List<Layer>();

        public NeuralNetwork(TopologyNetwork topology)
        {
            Topology = topology;

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputsCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Outputs).First();
            }
        }

        public double Learn(List<Tuple<double, double[]>> dataset, int epoch)
        {
            var error = 0.0;

            for (int i = 0; i < epoch; i++)
            {
                foreach (var data in dataset)
                {
                    error += Backpropagation(data.Item1, data.Item2);
                }
            }

            var result = error / epoch;
            return result;
        }

        private double Backpropagation(double exprected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Outputs;

            var difference = actual - exprected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            return Math.Pow(difference,2);
        }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                Layer layer = Layers[i];
                double[] previousLayerSingals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.ProcessingSignals(previousLayerSingals);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                double[] signal = new double[] { inputSignals[i] };
                Neuron neuron = Layers[0].Neurons[i];

                neuron.ProcessingSignals(signal);
            }
        }

        private void CreateOutputLayer()
        {
            Neuron[] outputNeurons = new Neuron[Topology.OutputsCount];
            Layer lastLayer = Layers.Last();

            for (int i = 0; i < Topology.OutputsCount; i++)
            {
                outputNeurons[i] = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
            }

            Layer outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology.HiddensLayers.Length; j++)
            {
                Layer lastLayer = Layers.Last();
                Neuron[] hiddenNeurons = new Neuron[Topology.HiddensLayers[j]];

                for (int i = 0; i < Topology.HiddensLayers[j]; i++)
                {
                    hiddenNeurons[i] = new Neuron(lastLayer.NeuronCount);
                }
                Layer hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateInputLayer(int countInputs = 1)
        {
            Neuron[] inputNeurons = new Neuron[Topology.InputsCount];
            for (int i = 0; i < Topology.InputsCount; i++)
            {
                inputNeurons[i] = new Neuron(countInputs, NeuronType.Input);
            }
            Layer inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
        static void Main() { }
    }
}
