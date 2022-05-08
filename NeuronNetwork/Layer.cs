namespace NeuralNetworks
{
    internal class Layer
    {
        public Neuron[] Neurons { get; private set; }
        public int Count => Neurons.Length;

        public Layer(Neuron[] neurons,NeuronType Classification = NeuronType.Normal)
        {
            Neurons = neurons;
        }

        public List<double> GetSignalsAtNeurons() 
        {
            List<double> SignalsNeurons = new List<double>();
            for (int i = 0; i < Neurons.Length; i++)
            {
                SignalsNeurons.Add(Neurons[i].Output);
            }
            return SignalsNeurons;
        }
    }
}
