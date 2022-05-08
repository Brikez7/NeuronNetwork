namespace NeuralNetworks
{
    class Neuron 
    {
        private List<double> Weight { get; } = new List<double>();
        private NeuronType Classification { get;}
        public double Output { get; private set; }
        public Neuron(int countWeight, NeuronType classification = NeuronType.Normal)
        {
            Classification = classification;

            Completion(countWeight);
        }
        public void SetWieght(params double[] weights) 
        {
            // DOTO: deleate become add possibility learning network
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = Weight[i];
            }
        }
        private void Completion(int countWeight) 
        {
            for (int i = 0; i < countWeight; i++)
            {
                Weight.Add(i);
            }
        }

        public double ProcessingSignal(List<double> inputsSignal) 
        {
            double sum = 0.0;
            for (int i = 0; i < inputsSignal.Count; i++)
            {
                sum += inputsSignal[i] * Weight[i];
            }
            Output = Sigmoid(sum);
            return Sigmoid(sum);
        }
        private double Sigmoid(double x) =>
            1.0 / (1.0 + Math.Pow(Math.E, -x));
        public override string ToString() => Output.ToString();
        
    }
}