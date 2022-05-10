namespace NeuralNetworks
{
    public class Layer
    {
        public Neuron[] Neurons { get; set; }
        public int CountNeurons => Neurons.Length;
        public NeuronType ClassificationLayer { get; private set; }
        public Layer(Neuron[] neurons,NeuronType classificationLayer = NeuronType.Normal)
        {
            Neurons = neurons;
            ClassificationLayer = classificationLayer;
        }

        public double[] GetSignals() 
        {
            double [] SignalsNeurons = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                SignalsNeurons[i] = Neurons[i].Output;
            }
            return SignalsNeurons;
        }

        public NeuronType GetTypeLayer() 
        {
            return ClassificationLayer;
        }
    }
}
