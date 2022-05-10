namespace NeuralNetworks
{
    public class Neuron 
    {
        public double[] Weight { get; private set; } 
        private double[] InputsSignals { get; set; } 
        private NeuronType Classification { get;}
        public double Delta { get; private set; }
        public double Output { get; private set; }
        public Neuron(int countWeight, NeuronType classification = NeuronType.Normal)
        {
            Classification = classification;
            InputsSignals = new double[countWeight];


            Weight = new double[countWeight];
            CompletionWeights(countWeight);
        }

        private void CompletionWeights(int countWeight)
        {
            if (Classification == NeuronType.input)
            {
                Weight = Weight.Select(x => x = 1).ToArray();
            }
            else
            {
                Random random = new Random();
                Weight = Weight.Select(x => x = random.NextDouble()).ToArray();
            }
        }

        public double ProcessingSignal(params double[] inputsSignals) 
        {
            InputsSignals = inputsSignals;

            double sum = 0.0;
            for (int i = 0; i < inputsSignals.Length; i++)
            {
                sum += InputsSignals[i] * Weight[i];
            }

            if (Classification != NeuronType.input)
            {
                Output = Sigmoid(sum);
            }
            else 
            {
                Output = sum;
            }
            return Output;
        }
        private double Sigmoid(double x) =>
            1.0 / (1.0 + Math.Pow(Math.E, -x));
        private double SigmoidDx(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid / (1.0 - sigmoid); 
        }
        public void Learn(double error, double learningRate)
        {
            if (NeuronType.input == Classification)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);
            for (int i = 0; i < Weight.Length; i++)
            {
                Weight[i] = Weight[i] - InputsSignals[i] * Delta * learningRate;
            }
        }
        public override string ToString() => Output.ToString();
    }
}