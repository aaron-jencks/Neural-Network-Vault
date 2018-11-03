using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkFundamentals.Math
{
    /// <summary>
    /// An interface for creating custom matrix types
    /// </summary>
    public interface IMatrix
    {
        /// <summary>
        /// Computes the inversion of the matrix and returns it as a new instance
        /// </summary>
        /// <returns>Returns the inverse of the matrix, if possible</returns>
        IMatrix Invert();

        /// <summary>
        /// Finds the determinant of the matrix
        /// </summary>
        /// <returns>Returns the determinant of the matrix</returns>
        double Det();
    }
}
