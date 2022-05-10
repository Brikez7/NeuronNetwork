using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetworks;
using System;
using System.Collections.Generic;

namespace NeuralNetworks.Tests
{
    [TestClass()]
    public class NerualNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            List<Tuple<double, double[]>> dataSet = new List<Tuple<double, double[]>>()
            {
                // Результат - Пациент болен - 1
                //             Пациент Здоров - 0

                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                //                                             T  A  S  F
                new Tuple<double, double[]> (0, new double[] { 0, 0, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 0, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 0, 0, 1, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 0, 1, 1 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 0, 1, 1, 0 }),
                new Tuple<double, double[]> (0, new double[] { 0, 1, 1, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 0, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 1, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 0, 1, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 0, 0 }),
                new Tuple<double, double[]> (0, new double[] { 1, 1, 0, 1 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 1, 0 }),
                new Tuple<double, double[]> (1, new double[] { 1, 1, 1, 1 })
            };
            int inputNeurons = 4;
            int outputNeurons = 1;
            int[] hiddenNeurons = new int[] { 2 };
            double errorRange = 0.01;
            TopologeNetwork topologe = new TopologeNetwork(inputNeurons, outputNeurons, hiddenNeurons, errorRange);
            NerualNetwork nerualNetwork = new NerualNetwork(topologe);

            double differeneDataSet = nerualNetwork.Learn(dataSet , 100000);
            
            double[] results = new double[dataSet.Count];

            for (int i = 0; i < results.Length; i++) 
            {
                results[i] = nerualNetwork.FeedForward(dataSet[i].Item2).Output;
            }

            Console.WriteLine(differeneDataSet);
            for (int i = 0; i < results.Length; i++)
            {
                double expected = Math.Round(dataSet[i].Item1,4);
                double expected2 = Math.Round(results[i], 4);
                Console.WriteLine("это "+ expected2 + " должно быть ровно этому"+ expected);
                //Assert.AreEqual(expected, expected);
            }
        }
    }
}