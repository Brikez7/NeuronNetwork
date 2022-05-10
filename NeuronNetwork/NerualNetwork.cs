namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        public TopologyNetwork Topology { get; private set; }
        public List<Layer> Layers { get; } = new List<Layer>();

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
            LayerProcessingSignals();

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
            double error = 0.0;

            for (int i = 0; i < epoch; i++)
            {
                foreach (var data in dataset)
                {
                    error += BackPropagation(data.Item1, data.Item2);
                }
            }

            return error / epoch;
        }

        private double BackPropagation(double exprected, params double[] inputs)
        {
            double actual = FeedForward(inputs).Outputs;

            double difference = actual - exprected;

            foreach (Neuron neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                Layer layer = Layers[j];
                Layer previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    Neuron neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        Neuron previousNeuron = previousLayer.Neurons[k];
                        double error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            return Math.Pow(difference,2);
        }

        private void LayerProcessingSignals()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                Layer layer = Layers[i];
                double[] previousLayerSingals = Layers[i - 1].GetSignals();

                foreach (Neuron neuron in layer.Neurons)
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
