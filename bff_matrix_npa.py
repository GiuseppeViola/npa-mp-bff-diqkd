import cvxpy as cp
from itertools import product
import chaospy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
np.bool = np.bool_  # Re-create the deprecated alias


def npa_solve(number_operators, constraint_task_list):
    """
    Solve an NPA hierarchy optimization problem.
    
    Returns the optimal value of the objective function
    """

    # Generate operator combinations for the NPA hierarchy
    combinations_list = generate_combinations(number_operators, LEVEL) 
    combinations_list = apply_commutation(combinations_list)
    combinations_list = apply_idempotency(combinations_list)
    combinations_list = apply_orthogonality_bff(combinations_list)
    combinations_list_length = len(combinations_list)
    print('Moment matrix dimension: ', combinations_list_length)
    # Create moment matrix variable
    matrix_dimension = combinations_list_length * DIMENSION
    if not REAL_OBJECTIVE:
        gamma = cp.Variable((matrix_dimension, matrix_dimension), hermitian = True) 
    else:
        gamma = cp.Variable((matrix_dimension, matrix_dimension), symmetric = True) 

    # Map each operator word (i.e. word_i * word_j^dagger) to the block
    # indices (i, j) of gamma, then simplify duplicate words that reduce
    # to the same word under commutation/idempotency.
    words_dictionary = generate_words_dictionary_bff(combinations_list) 
    words_dictionary_simplified = transform_dictionary_commutation(words_dictionary)
    words_dictionary_simplified = transform_dictionary_idempotency(words_dictionary_simplified)
    
    constraints_orthogonality = generate_constraint_orthogonality_matrix_bff(gamma,words_dictionary_simplified)
    constrains_same_words = extract_equalities_matrix(words_dictionary_simplified, gamma)
    trace_rho = sum([gamma[i,i] for i in range(DIMENSION)])
    
    # Grid of CHSH scores to sweep over, avoiding the exact endpoints
    # (where the entropy bound is singular / the SDP can become
    # numerically ill-conditioned).
    eps = 1e-3
    scores = np.linspace(WMIN+eps, WMAX-eps, NPOINTS)

    constraints0 = [trace_rho == 1, gamma >> 0]  + constrains_same_words + constraints_orthogonality
    result_list = []
    for score in scores:
        result = 0
        # Gauss-Radau quadrature sum approximating the conditional
        # von Neumann entropy: each quadrature node T[k] with weight
        # W[k] contributes one SDP sub-problem, following the
        # Brown-Fawzi-Fawzi (BFF) entropy-estimation method.
        for k in range(len(T)):
            ck = W[k]/(T[k] * np.log(2))
            constraints = constraints0
            for j in range(len(constraint_task_list)):
                # Convert the raw CHSH expression into Collins-Gisin
                # (CG) form, then into the moment-matrix objective.
                constraint_list_CG = convert_to_CG_matrix(constraint_task_list[j])
                constraint_expression =   generate_objective_function_matrix(constraint_list_CG, gamma,words_dictionary)   
                constraints = constraints + [constraint_expression >= score]
            if TRUSTED_PARTY == 'A':
                objective_list = objective_list_bff_matrix_A(k)
            elif TRUSTED_PARTY == 'B':
                objective_list = objective_list_bff_matrix_B(k)

            objective_list_CG = convert_to_CG_matrix(objective_list)
            objective_function = generate_objective_function_matrix(objective_list_CG, gamma,words_dictionary)
            if not REAL_OBJECTIVE:
                objective_function = cp.real(objective_function)
            problem = cp.Problem(cp.Minimize(objective_function), constraints) # create the problem object
            problem.solve(verbose = False, solver = SOLVER) # solve the optimization problem
            if problem.status == 'optimal':
                # Accumulate this quadrature node's contribution to the
                # entropy bound / key rate.
                result += float(ck * (1 + problem.value))
            else:
                # If any node fails to solve optimally, discard the
                # whole score point.
                result = 0
                print('Not optimal result found: ', problem.value)
                break
        
        # --- CHSH reference curve -----------------------------------
        # Hvn(score) is the known closed-form von Neumann entropy bound
        # for the CHSH protocol, used here as a sanity-check reference
        # against the numerically computed `result`.
        print([score, float(Hvn(score)), result])
        result_list.append([score, Hvn(score), result])
        
        # --- BB84 reference curve -----------------------------------
        # print([score, 1-float(h(score)), result])
        # result_list.append([score, 1-h(score), result])
    
    return result_list



def h(p):
    """Binary Shannon entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    return -p*np.log2(p) - (1-p)*np.log2(1-p)


def Hvn(w):
    """
    Closed-form lower bound on the conditional von Neumann entropy for
    the CHSH protocol, as a function of the observed CHSH score `w`.
    """
    return 1 - h(1/2 + np.sqrt((8*w - 4)**2 / 4 - 1)/2)


def generate_combinations(number_operators, npa_level):
    """
    Generate all operator combinations up to a given NPA level.
            
    Examples:
        
    >>> generate_combinations(2, 2)
    [[], [0], [1], [0, 0], [0, 1], [1, 0], [1, 1]]
    
    >>> generate_combinations(3, 1)
    [[], [0], [1], [2]]
    """
    operators_list = [[]]  # Identity
    for level in range(1, npa_level + 1):
        operators_list.extend([list(comb) for comb in product(range(number_operators), repeat=level)])
    return operators_list


def apply_commutation(combinations_list):
    """
    Reduce each operator word to its canonical form under the
    commutation relations in COMMUTING_OPERATORS, deduplicating the
    resulting list of words (order-preserving, first occurrence kept).
    """
    new_combinations = []
    for combination in combinations_list:
        new_combination = transform_key_commutation(combination)
        if new_combination not in new_combinations:
            new_combinations.append(new_combination)
    return new_combinations


def apply_idempotency(combinations_list):
    """
    Reduce each operator word by collapsing consecutive duplicate
    idempotent operators (P*P = P), deduplicating the resulting list.
    """
    new_combinations = []
    for combination in combinations_list:
        new_combination = transform_key_idempotency(combination)
        if new_combination not in new_combinations:
            new_combinations.append(new_combination)
    return new_combinations


def apply_orthogonality_bff(combinations_list):
    """
    Drop any operator word that is identically zero because it contains
    two adjacent, distinct operators belonging to the same measurement
    (orthogonal projectors), as determined by `check_key_orthogonal_bff`.
    """
    new_combinations = []
    for combination in combinations_list:
        if check_key_orthogonal_bff(combination):
            continue
        new_combinations.append(combination)
    return new_combinations


def transform_key_commutation(key_list):
    """
    Reorder the operators in `key_list` so that operators belonging to
    the same commuting group (as defined in COMMUTING_OPERATORS) are
    grouped together, while operators not
    belonging to any group keep their original relative order at the
    end. This yields a canonical form for words equal under the
    scenario's commutation relations.
    """
    result = []
    used_positions = set()
    
    # Add elements from each list in order
    for group in COMMUTING_OPERATORS:
        group_set = set(group)
        for pos, idx in enumerate(key_list):
            if idx in group_set and pos not in used_positions:
                result.append(idx)
                used_positions.add(pos)
    
    # Add remaining elements not in any list
    for pos, idx in enumerate(key_list):
        if pos not in used_positions:
            result.append(idx)
    return result


def transform_dictionary_commutation(dictionary):
    """
    Rewrite the keys of `dictionary` (operator words) into their
    canonical commutation form, merging the value lists of any keys
    that collapse onto the same canonical word.
    """    
    result = defaultdict(list)
    for key, value in dictionary.items():
        new_key = tuple(transform_key_commutation(key))
        result[new_key].extend(value)  # extend instead of append
    return dict(result)


def transform_dictionary_idempotency(dictionary):
    """
    Rewrite the keys of `dictionary` (operator words) by collapsing
    consecutive idempotent duplicates, merging value lists of keys that
    collapse onto the same reduced word.
    """
    result = defaultdict(list)
    for key, value in dictionary.items():
        new_key = tuple(transform_key_idempotency(key))
        result[new_key].extend(value)  # extend instead of append
    return dict(result)

def transform_key_idempotency(key):
    """
    Remove consecutive duplicates of operators in IDEMPOTENT_OPERATORS
    (i.e. apply P*P = P for projective measurement operators).
    """
    if len(key) == 0:
        return key
    
    result = [key[0]]
    
    for i in range(1, len(key)):
        if key[i] == key[i-1] and key[i] in IDEMPOTENT_OPERATORS:
            continue
        result.append(key[i]) 
    return result


def generate_constraint_orthogonality_matrix_bff(gamma, words_dictionary):
    """
    Build the list of SDP equality constraints setting to zero every
    block of `gamma` corresponding to a word that is identically zero
    due to orthogonality of same-measurement outcomes.
    """    
    constraints = []
    for key in words_dictionary:
        if check_key_orthogonal_bff(key):
            i, j = words_dictionary[key][0]
            for i_inner in range(DIMENSION):
                for j_inner in range(DIMENSION):
                    constraints.append(gamma[i*DIMENSION+i_inner,j*DIMENSION+j_inner] == 0)
    return constraints


def check_key_orthogonal_bff(key):
    """
    Determine whether an operator word is identically zero because it
    contains two adjacent, distinct operators that belong to the same
    measurement (and are therefore orthogonal projectors, P_a * P_b = 0
    for a != b).
    """
    index_to_op = {}
    idx = 0
    for i, op_list in enumerate(SCENARIO_CG):
        if i < 1:
            for j, val in enumerate(op_list):
                for _ in range(val):
                    index_to_op[idx] = (i, j)
                    idx += 1
    
    for i in range(len(key) - 1):
        if key[i] != key[i+1] and key[i] in index_to_op and key[i+1] in index_to_op:
            if index_to_op[key[i]] == index_to_op[key[i+1]]:
                return True
    return False


def extract_equalities_matrix(dictionary, gamma):
    """
    Build SDP equality constraints connecting together every pair of gamma
    blocks that correspond to the same reduced operator word.
    """    
    equalities = []
    for pairs in dictionary.values():
        if len(pairs) < 2:
            continue  # no equality to extract

        i0, j0 = pairs[0]
        for i, j in pairs[1:]:
            for i_inner in range(DIMENSION):
                for j_inner in range(DIMENSION):
                    equalities.append(gamma[i0*DIMENSION+i_inner,j0*DIMENSION+j_inner]== gamma[i*DIMENSION+i_inner,j*DIMENSION+j_inner])
    return equalities


def convert_to_CG_matrix(expressions):
    """
    Expand a list of expressions [(coef, tuple_of_indices)] according
    to SCENARIO and SCENARIO_CG rules.
    """
    scenario_cg_list = sum([inner_list for inner_list in SCENARIO_CG], [])
    scenario_indexes = generate_scenario_mapping(SCENARIO)
    scenario_cg_indexes = generate_scenario_mapping(SCENARIO_CG)
    
    NEW_EXPRESSIONS = []
    def recursive_expansion_matrix(coef, matrix, left, right):
        if len(right) == 0:
            # Base case: no more elements to process
            NEW_EXPRESSIONS.append([coef, matrix, left])
            return
        
        elem = right[0]
        j = find_index_containing(scenario_indexes, elem)
        position = scenario_indexes[j].index(elem)
        
        if position < scenario_cg_list[j]:
            converted_idx = scenario_cg_indexes[j][position]
            recursive_expansion_matrix(coef, matrix, left + [converted_idx], right[1:])
            return
        
        # Decomposition needed
        # First: remove the element (don't add it to left)
        recursive_expansion_matrix(coef, matrix, left, right[1:])
        
        # Then: add each replacement with flipped sign
        for i in range(scenario_cg_list[j]):
            recursive_expansion_matrix(-coef, matrix, left + [scenario_cg_indexes[j][i]], right[1:])
    
    # Process each expression
    for coef, matrix, elem in expressions:
        right = elem
        left = []
        recursive_expansion_matrix(coef, matrix, left, right)
    return NEW_EXPRESSIONS


def generate_scenario_mapping(scenario):
    """
    Assign a contiguous block of sequential indices to each integer
    entry of a nested scenario description.
    """
    mapping = []
    current_index = 0
    
    for element in scenario:
        for integer in element:
            # Create a list of indices for this integer
            indices = list(range(current_index, current_index + integer))
            mapping.append(indices)
            current_index += integer
    return mapping


def find_index_containing(lst, target):
    """
    Return the index of the first sublist in `lst` that contains `target`, or None if not found.
    """
    for i, elem in enumerate(lst):
        if target in elem:
            return i
    return None  # None if not found


def generate_objective_function_matrix(objective_list, gamma, words_dictionary):
    """
    Construct a linear (in gamma) objective/constraint expression as a
    weighted sum of moment-matrix blocks.
    
    Each term is specified by a scalar coefficient, a fixed
    DIMENSION x DIMENSION matrix, and an operator word identifying which block of
    gamma to use. The term's contribution is coefficient * Tr(matrix @ gamma_block).
    """   
    total = 0
    for coefficient, matrix, word in objective_list:
        i, j = words_dictionary[tuple(word)][0]
        matrix_gamma = [[gamma[i*DIMENSION+i_inner,j*DIMENSION+j_inner]for j_inner in range(DIMENSION)] for i_inner in range(DIMENSION)]
        total += coefficient* trace(matrix_multiply(matrix, matrix_gamma))      
    return total


def matrix_multiply(A, B):
    """
    Multiply two square matrices given as nested Python lists.
    """
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def trace(A):
    """Sum of the diagonal entries of a square matrix given as a nested list."""
    n = len(A)
    result = A[0][0]
    for i in range(1, n):
        result = result + A[i][i]
    return result


""" Constraints preparation """

def to_base_10(digits, bases):
    """Convert mixed-radix digits to an integer."""
    if len(digits) != len(bases):
        raise ValueError("digits and bases must have the same length")

    value = 0
    multiplier = 1
    for digit, base in zip(reversed(digits), reversed(bases)):
        value += digit * multiplier
        multiplier *= base
    return value


def generate_commuting_groups(scenario):
    """
    Partition operator indices into commuting groups, one group per
    party, each containing sequential indices for all operators belonging to that
    party.
    """
    result = []
    counter = 0
    for t in scenario:
        # Calculate sum of integers in current tuple
        n = sum(t)
        # Generate n progressive integers
        progressive_list = list(range(counter, counter + n))
        result.append(progressive_list)
        # Update counter for next tuple
        counter += n 
    return result


def zx_basis_projectors():
    """
    Build the four rank-1 projectors used for the trusted party's Z and
    X basis measurements on a qubit, each represented as a 2x2 matrix (nested list).
    """
    projectors=[]
    R0=[[1,0],[0,0]]
    R1=[[1/2,1/2],[1/2,1/2]]
    ident =[
        [1, 0],
        [0, 1],
        ]
    projectors.append(R0)
    projectors.append(sum_matrices([negate_matrix(R0),ident]))
    projectors.append(R1)
    projectors.append(sum_matrices([negate_matrix(R1),ident]))
    return projectors


def sum_matrices(b):
    """Element-wise sum of a list of equal-sized square matrices (nested lists)."""
    N = len(b[0])  # Dimension of each square list
    result = [[0] * N for _ in range(N)]  # Initialize result matrix with zeros
    for matrix in b:
        for i in range(N):
            for j in range(N):
                result[i][j] += matrix[i][j]  # Element-wise addition
    return result

def negate_matrix(matrix):
    """Return the element-wise negation of a square matrix (nested list)."""
    N = len(matrix)  # Assuming matrix is square of size N x N
    return [[-matrix[i][j] for j in range(N)] for i in range(N)]


""" bff matrix form """


def objective_list_bff_matrix_A(k):
    """
    Build the k-th Gauss-Radau quadrature-node objective function for the
    Brown-Fawzi-Fawzi (BFF) entropy bound, applying the matrix NPA method
    for the variant where Alice is the trusted and generating key party.
    """
    objective_function_list = []
    identity = np.eye(DIMENSION)
    x0 = 0
    for a in range(nA):
        iA = to_base_10([x0, a], [nX, nA])
        iZ = nX*nA + to_base_10([0, a], [1, nA])
        iZ_star = iZ + nA

        element = [iZ]
        objective_function_list.append([1, TRUSTED_PROJECTORS[iA], element])
        
        element = [iZ_star]
        objective_function_list.append([1, TRUSTED_PROJECTORS[iA], element])
        
        element = [ iZ_star, iZ]
        objective_function_list.append([1-T[k], TRUSTED_PROJECTORS[iA], element])
        
        element = [iZ, iZ_star]
        objective_function_list.append([T[k], identity, element])  
    return(objective_function_list)


def objective_list_bff_matrix_B(k):
    """
    Build the k-th Gauss-Radau quadrature-node objective function for the
    Brown-Fawzi-Fawzi (BFF) entropy bound, applying the matrix NPA method
    for the variant where Bob is the trusted party and Alice is the key generating party.
    """
    objective_function_list = []
    identity = np.eye(DIMENSION)
    x0 = 0
    for a in range(nA):
        iA = to_base_10([x0, a], [nX, nA])
        iZ = nX*nA + to_base_10([0, a], [1, nA])
        iZ_star = nX*nA + nA + to_base_10([0, a], [1, nA])

        element = [iA, iZ]
        objective_function_list.append([1, identity, element])
        
        element = [iA, iZ_star]
        objective_function_list.append([1, identity, element])
        
        element = [iA, iZ_star, iZ]
        objective_function_list.append([1-T[k], identity, element])
        
        element = [iZ, iZ_star]
        objective_function_list.append([T[k], identity, element])
    return(objective_function_list)


def chsh_game_form_matrix(nX, nA):
    """
    Build the CHSH winning-probability expression, for the matrix NPA method
    in the in which one party is trusted (Alice or Bob) and the other is not.
    """
    objective_function_list = []
    coefficient = 1/(nX**2)
    
    for x in range(nX):
        for a in range(nA):
            index = to_base_10([x, a], [nX, nA])
            matrix = TRUSTED_PROJECTORS[index]
            for y in range(nX):
                for b in range(nA):
                    if  x * y == (a ^ b):
                        iB = to_base_10([y, b], [nX, nA])
                        word = [iB]
                        objective_function_list += [[coefficient,matrix,word]]
    return [objective_function_list]


def bb84_matrix(nX, nA):
    """
    Build the BB84 success-probability expression, for the matrix NPA method
    in the in which one party is trusted (Alice or Bob) and the other is not.
    """
    coefficient = 1
    objective_function_list_all_x = []
    for x in range(nX):
        objective_function_list = []
        for a in range(nA):
            index = to_base_10([x, a], [nX, nA])
            matrix = TRUSTED_PROJECTORS[index]
            iB = to_base_10([x, a], [nX, nA])
            word = [iB]
            objective_function_list += [[coefficient,matrix,word]]
        objective_function_list_all_x.append(objective_function_list)
    return objective_function_list_all_x


def generate_quadrature(m):
    """Generate Radau quadrature nodes and weights on [0, 1]."""
    t, w = chaospy.quadrature.radau(m, chaospy.Uniform(0, 1), 1)
    t = t[0]
    return t, w


def generate_words_dictionary_bff(combinations_list):
    """
    Build the dictionary mapping each NPA operator word
    (i.e. word_i concatenated with the reverse of word_j) to the list of (i, j) block-index
    pairs of the moment matrix that realize that word. Only the first occurrence of each word is stored.
    """
    words_dictionary = {}
    for i in range(len(combinations_list)):
        for j in range(len(combinations_list)):
            # Concatenate sequence i with reversed sequence j
            key_bff = combinations_list[i] + swap_Z_Zstar(combinations_list[j][::-1])
            key = tuple(key_bff)
            words_dictionary.setdefault(key, []).append((i, j))
    return words_dictionary


def swap_Z_Zstar(combinations_list):
    """    
    Swap indices representing the BFF "Z" operators with their
    corresponding "Z*" (Hermitian conjugate) operators and vice versa.
    """
    result = []
    for i in combinations_list:
        if nX * nA_CG <= i <= nX * nA_CG + nA - 1:
            result.append(i + nA)
        elif nX * nA_CG + nA <= i <= nX * nA_CG + 2 * nA - 1:
            result.append(i - nA)
        else:
            result.append(i)
    return result


# --- Protocol / scenario configuration ---------------------------------


TRUSTED_PARTY = 'B'
SCENARIO = [[2, 2], [4]]
NPOINTS = 20
LEVEL = 2
SOLVER = "MOSEK"
WMIN = 0.75
WMAX = 0.85
M = 4       # Number of Gauss-Radau quadrature nodes / 2, for the BFF entropy approximation.
TRUSTED_PROJECTORS = zx_basis_projectors()


SCENARIO_CG = [[x - 1 for x in SCENARIO[0]], *SCENARIO[1:]]
REAL_OBJECTIVE = True
# number of inputs
nX = len(SCENARIO[0])
# number of outcomes
nA = SCENARIO[0][0]
nA_CG = nA-1
DIMENSION = len(TRUSTED_PROJECTORS[0][0])
T, W = generate_quadrature(M)
COMMUTING_OPERATORS = generate_commuting_groups(SCENARIO_CG)
IDEMPOTENT_OPERATORS = list(range(sum(sum(t) for t in [SCENARIO_CG[0]])))

constraint_task_list = chsh_game_form_matrix(nX, nA)
# constraint_task_list = bb84_matrix(nX, nA)

number_operators = sum(element for subtuple in SCENARIO_CG for element in subtuple)
result = npa_solve(number_operators, constraint_task_list)
result_array = np.array(result)  # Convert to numpy array
x = result_array[:, 0]
y = result_array[:, 1]  
z = result_array[:, 2]

plt.plot(x, z, marker='o', label=f'bff, {TRUSTED_PARTY} trusted')
plt.ylim(0.0, 1.0)
plt.xlim(WMIN, WMAX)
plt.legend()
plt.show()

data = np.column_stack((x, y, z))
OUTPUT_FILE = "plot_data.txt"
np.savetxt(
    OUTPUT_FILE,
    data,
    header="x y z",
    comments=""
)