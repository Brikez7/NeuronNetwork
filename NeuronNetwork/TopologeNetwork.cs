namespace NeuralNetworks
{
    public class TopologeNetwork
    {
        public int InputsCount { get; }
        public int OutputsCount { get; }
        public double LearningRate { get; }
        public int[] HiddensLayers { get; }

        public TopologeNetwork(int inputCount, int outputCount, double learningRate, params int[] hiddensLayers)
        {
            InputsCount = inputCount;
            OutputsCount = outputCount;
            LearningRate = learningRate;
            HiddensLayers = hiddensLayers;
        }
    }
}
