namespace NeuralNetworks
{
    public class Layer
    {
        public Neuron[] Neurons { get; }
        public int NeuronCount => Neurons?.Length ?? 0;
        public NeuronType ClassificationLayer;

        public Layer(Neuron[] neurons, NeuronType type = NeuronType.Normal)
        {
            // TODO: проверить все входные нейроны на соответствие типу

            Neurons = neurons;
            ClassificationLayer = type;
        }

        public double[] GetSignals()
        {
            double [] signals = new double[Neurons.Length]; 
            for(int i = 0; i < Neurons.Length; i++)
            {
                signals[i] = Neurons[i].Output;
            }
            return signals;
        }

        public override string ToString()
            => ClassificationLayer.ToString();

    }
}
