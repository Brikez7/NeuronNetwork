namespace NeuralNetworks
{
    internal class TopologeNetwork
    {
        public int CountInputs { get; private set; }
        public int CountOutputs { get; private set; }
        public int[] HiddingLayers { get; private set; }

        public TopologeNetwork(int countInputs, int countOutputs,params int[] hiddingLayers)
        {
            CountInputs = countInputs;
            CountOutputs = countOutputs;
            HiddingLayers = hiddingLayers;
        }
    }
}
