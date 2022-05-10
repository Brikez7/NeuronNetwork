namespace NeuralNetworks
{
    public class Neuron
    {
        public double[] Weights { get; private set; }
        public double[] Inputs { get; private set; }
        public NeuronType ClassificationNeuron { get; private set; }
        public double Outputs { get; private set; }
        public double Delta { get; private set; }

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            ClassificationNeuron = type;
            Inputs = new double[inputCount];
            Weights = new double[inputCount];

            FillWeights(inputCount);
        }

        private void FillWeights(int inputCount)
        {
            if (ClassificationNeuron == NeuronType.Input)
            {
                Weights =  Weights.Select(x => x = 1).ToArray();
            }
            else
            {
                Random random = new Random();
                Weights = Weights.Select(x => x = random.NextDouble()).ToArray();
            }
        }

        public double ProcessingSignals(double[] inputSignals)
        {
            Inputs = inputSignals;

            double sum = 0.0;
            for (int i = 0; i < inputSignals.Length; i++)
            {
                sum += Inputs[i] * Weights[i];
            }

            if (ClassificationNeuron != NeuronType.Input)
            {
                Outputs = Sigmoid(sum);
            }
            else
            {
                Outputs = sum;
            }

            return Outputs;
        }

        private double Sigmoid(double x)
            => 1.0 / (1.0 + Math.Pow(Math.E, -x));

        private double SigmoidDx(double x)
        {
            double sigmoid = Sigmoid(x);
            return sigmoid / (1 - sigmoid);
        }

        public void Learn(double error, double learningRate)
        {
            if (ClassificationNeuron == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Outputs);

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Weights[i] - Inputs[i] * Delta * learningRate;
            }
        }

        public override string ToString()
            => Outputs.ToString();
    }
}
