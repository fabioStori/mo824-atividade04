package problems.qbf;

import solutions.Solution;

import java.io.IOException;

/**
 * Class representing the inverse of the Quadractic Binary Function
 * ({@link KQBF}), which is used since the GRASP is set by
 * default as a minimization procedure.
 * 
 * @author ccavellucci, fusberti
 */
public class KQBF_Inverse extends KQBF {

	/**
	 * Constructor for the QBF_Inverse class.
	 * 
	 * @param filename
	 *            Name of the file for which the objective function parameters
	 *            should be read.
	 * @throws IOException
	 *             Necessary for I/O operations.
	 */
	public KQBF_Inverse(String filename) throws IOException {
		super(filename);
	}


	/* (non-Javadoc)
	 * @see problems.qbf.QBF#evaluate()
	 */
	@Override
	public Double evaluateQBF(Solution<Integer> sol) {
		return -super.evaluateQBF(sol);
	}
	
	/* (non-Javadoc)
	 * @see problems.qbf.QBF#evaluateInsertion(int)
	 */
	@Override
	public Double evaluateInsertionQBF(int i) {	
		return -super.evaluateInsertionQBF(i);
	}
	
	/* (non-Javadoc)
	 * @see problems.qbf.QBF#evaluateRemoval(int)
	 */
	@Override
	public Double evaluateRemovalQBF(int i) {
		return -super.evaluateRemovalQBF(i);
	}
	
	/* (non-Javadoc)
	 * @see problems.qbf.QBF#evaluateExchange(int, int)
	 */
	@Override
	public Double evaluateExchangeQBF(int in, int out) {
		return -super.evaluateExchangeQBF(in,out);
	}

}
