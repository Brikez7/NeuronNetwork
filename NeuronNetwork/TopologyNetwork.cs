namespace NeuralNetworks
{
    public class TopologyNetwork
    {
        public int InputsCount { get; private set; }
        public int OutputsCount { get; private set; }
        public double LearningRate { get; private set; }
        public int[] HiddensLayers { get; private set; }

        public TopologyNetwork(int inputCount, int outputCount, double learningRate, params int[] hiddensLayers)
        {
            InputsCount = inputCount;
            OutputsCount = outputCount;
            LearningRate = learningRate;
            HiddensLayers = hiddensLayers;
        }
    }
}
