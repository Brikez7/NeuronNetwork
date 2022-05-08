    
namespace NeuralNetworks
{
    internal class NerualNetwork
    {
        private List<Layer> Layers { get; }
        private TopologeNetwork Topologe{ get; }

        public NerualNetwork(TopologeNetwork topologe)
        {
            Topologe = topologe;
            CreateInputLayers();
            CreateOutputLayers();
            CreateHiddenLayers();
        }

        private void CreateInputLayers(int CountInputsNeuron = 1) 
        {
            List<Neuron> inputsNeurons = new List<Neuron>();
            for (int i = 0; i < Topologe.CountInputs; i++)
            {
                Neuron neuron = new Neuron(CountInputsNeuron, NeuronType.input);
                inputsNeurons.Add(neuron);
            }
            Layer inputlayer = new Layer(inputsNeurons.ToArray(),NeuronType.input);
            Layers.Add(inputlayer);
        }
        private void CreateOutputLayers()
        {
            List<Neuron> outputsNeurons = new List<Neuron>();
            Layer lastLayer = Layers.Last();
            for (int i = 0; i < Topologe.CountInputs; i++)
            {
                Neuron neuron = new Neuron(lastLayer.Count, NeuronType.output);
                outputsNeurons.Add(neuron);
            }
            Layer inputLayer = new Layer(outputsNeurons.ToArray(), NeuronType.output);
            Layers.Add(inputLayer);
        }
        private void CreateHiddenLayers()
        {
            for (int i = 0; i < Topologe.HiddingLayers.Length; i++)
            {
                List<Neuron> hiddenNeurons = new List<Neuron>();
                Layer lastLayer = Layers.Last();
                for (int ii = 0; ii < Topologe.CountInputs; ii++)
                {
                    Neuron neuron = new Neuron(lastLayer.Count);
                    hiddenNeurons.Add(neuron);
                }
                Layer inputlayer = new Layer(hiddenNeurons.ToArray(), NeuronType.Normal);
                Layers.Add(inputlayer);
            }
        }

        private Neuron FeedForward(List<List<double>> InputsSignalsAtNeuron)
        {
            SendSignalsOnInputsNeurons(InputsSignalsAtNeuron);
            ProcessingSignals();
            if (Topologe.CountOutputs > 1)
            {
                return Layers.Last().Neurons.OrderBy(n => n.Output).FirstOrDefault();
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
                List<double> signals = Layers[i - 1].GetSignalsAtNeurons();
                foreach (var neuron in layer.Neurons)
                {
                    neuron.ProcessingSignal(signals);
                }
            }
        }

        private void SendSignalsOnInputsNeurons(List<List<double>> InputsSignalsAtNeuron)
        {
            for (int i = 0; i < InputsSignalsAtNeuron.Count; i++)
            {
                List<double> Signal = InputsSignalsAtNeuron[i];
                Neuron neuron = Layers[0].Neurons[i];

                neuron.ProcessingSignal(Signal);
            }
        }
    }
}
