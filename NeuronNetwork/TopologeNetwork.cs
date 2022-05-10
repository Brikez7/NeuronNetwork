namespace NeuralNetworks
{
    public class TopologeNetwork
    {
        public int CountInputs { get; private set; }
        public int CountOutputs { get; private set; }
        public int[] HiddingLayers { get; private set; }
        public double LearningRate { get; private set; }
        public TopologeNetwork(int countInputs, int countOutputs, int[] hiddingLayers, double errorRange)
        {
            CountInputs = countInputs;
            CountOutputs = countOutputs;
            HiddingLayers = hiddingLayers;
            LearningRate = errorRange;
        }
    }
}
