using Parameters, LinearAlgebra, PrettyTables, Plots, BenchmarkTools

"""
    Contains the Parameters used
"""

# Now you can redefine it with @with_kw
@with_kw mutable struct NeoclassicalGrowthModel
    # Steady state parameters
    β::Float64 = 0.96  # discount factor
    α::Float64 = 0.4   # capital share
    δ::Float64 = 0.1   # depreciation rate
    n::Int = 1         # number of grid points
    kgrid::Vector{Float64} = zeros(1)  # capital grid vector
    A::Float64 = 1.0 # tfp level
    kss::Float64 = 1.0 # ss level of capital 

    # Auxiliary parameters for capital grid
    ub::Float64 = 10.0 # upper bound of grid = 10 times k_ss; could be defined as int but I want to keep it flexible
    lb::Float64 = 0.7 #= lower bound of grid = 70% of k_ss level. This is a decline of 30% relative to SS level 
    similar to the decline in gdp observed in the great depression. 
    It's unlikely that we'd observe worse outcomes than this. =#

    # Auxiliary parameters for vfi exploiting monotonicity
    idx_initial::Int = 1

    # Auxiliary constant for MPI
    nbf_mpi::Int = 1 # number of brute force iterations before MPI algorithm kicks in

    # utility function 
    u = c -> log(c);
end

"""
    Function to create grid for capital
"""

function capitalgrid(NGM::NeoclassicalGrowthModel)
    @unpack n,β,α,A,δ,ub,lb = NGM;
    
    #Compute SS level of capital
    NGM.kss = (1/(A*α)*((1/β)+δ-1))^(1/(α-1));

    lower = lb*NGM.kss;
    upper = ub*NGM.kss;

    NGM.kgrid = LinRange(lower,upper,n);
    return NGM.kgrid
end

"""
    Function to solve VFI using a stopping rule based on value function convergence
"""

function vfi1(NGM::NeoclassicalGrowthModel,Vguess::Vector{Float64},tol::Float64)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α,β,δ,n,kgrid,u = NGM;

    # Initialize values for the interation 
    dif = 1;
    iter = 0;
    Vold = copy(Vguess);
    Vnew = copy(Vold);
    k′ = Vector{Float64}(undef, n);  # policy function for capital
    
    # vfi 
    while dif>tol
        Vold = copy(Vnew);
        for i = 1:n 
            vec = Vector{Float64}(undef, n);
            for j = 1:n # loop on k′
                # compute consumption 
                c = kgrid[i]^α + (1-δ)*kgrid[i] - kgrid[j]; # note that I'm assuming k′ lies on the same grid as k
                
                # compute utility 
                if c > 0
                    util = u(c);
                else
                    util = -Inf;
                end

                # vector to evaluate and choose optimal k′
                vec[j] = util + β*Vold[j];
            end

            # compute optimal choice for k′
            Vnew[i], idx = findmax(vec)
            k′[i] = kgrid[idx];
        end

        # Stopping rule based on value function convergence:
        dif = norm(Vnew-Vold,Inf);
        iter += 1;

        # Print iteration and difference between Vnew and Vold
        #println("Iteration $iter: Difference between Vnew and Vold = ", dif)
    end

    return Vnew, k′, iter
end

"""
    Function to solve VFI using a stopping rule based on value function convergence but without for loops (only matrices)
"""

function vfi1_matrix(NGM::NeoclassicalGrowthModel, Vguess::Vector{Float64}, tol::Float64)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, u = NGM
    Vold = copy(Vguess)
    Vnew = copy(Vguess)
    k′ = similar(kgrid)  # Policy function for capital
    dif = 1
    iter = 0
    
    C = kgrid.^α .+ (1 - δ) .* kgrid .- kgrid'  
    U = similar(C)
    U .= -Inf  # Initialize with -Inf
    mask = C .> 0
    U[mask] .= u.(C[mask])

    while dif > tol
        Vold = copy(Vnew)
    
        # Bellman update:
        Vmat = U .+ β .* Vold'
        Vnew, idx = findmax(Vmat,dims = 2)  # Get max value and optimal choice
        idx = [i[2] for i in idx]
        
        # Extract policy function
        k′ = kgrid[idx]  
    
        # Convergence check
        dif = norm(Vnew .- Vold, Inf)
        iter += 1
    end

    return vec(Vnew), vec(k′), iter
end

"""
    Function to solve VFI using a stopping rule based on policy function convergence
"""

function vfi2(NGM::NeoclassicalGrowthModel, Vguess::Vector{Float64}, tol::Float64)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, u = NGM;

    # Initialize values for the iteration 
    dif = 1;
    iter = 0;
    Vold = copy(Vguess);
    Vnew = copy(Vold);
    k′ = Vector{Float64}(undef, n);  # policy function for capital
    k′old = copy(k′);  # initialize previous policy function for convergence check
    
    # vfi 
    while dif > tol
        k′old = copy(k′)  # Update previous policy function
        Vold = copy(Vnew);
    
        for i = 1:n 
            vec = Vector{Float64}(undef, n);;
            
            for j = 1:n  # loop on k′
                # compute consumption 
                c = kgrid[i]^α + (1 - δ) * kgrid[i] - kgrid[j];  # assuming k′ lies on the same grid as k
                
                # compute utility 
                if c > 0
                    util = u(c);
                else
                    util = -Inf;
                end

                # vector to evaluate and choose optimal k′
                vec[j] = util + β * Vold[j];
            end

            # compute optimal choice for k′
            Vnew[i], idx = findmax(vec)
            k′[i] = kgrid[idx];
        end

        # Stopping rule based on policy function convergence:
        dif = norm(k′ - k′old, Inf);  # Convergence based on the change in the policy function        
        #dif = norm(Vnew - Vold, Inf);
        iter += 1;

        # Print iteration and difference between k′ and k′old
        #println("Iteration $iter: Policy function difference = ", dif)
    end

    return Vnew, k′, iter
end

"""
    Function to solve VFI using a stopping rule based on policy function convergence but without for loops (only matrices)
"""

function vfi2_matrix(NGM::NeoclassicalGrowthModel, Vguess::Vector{Float64}, tol::Float64)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, u = NGM
    Vold = copy(Vguess)
    Vnew = copy(Vguess)
    k′ = similar(kgrid)  
    kold = similar(kgrid)
    dif = 1
    iter = 0
    
    C = kgrid.^α .+ (1 - δ) .* kgrid .- kgrid'  
    U = similar(C)
    U .= -Inf  # Initialize with -Inf
    mask = C .> 0
    U[mask] .= u.(C[mask])

    while dif > tol
        kold = copy(k′)  # Update previous policy function
        Vold = copy(Vnew)
    
        # Bellman update:
        Vmat = U .+ β .* Vold'
        Vnew, idx = findmax(Vmat,dims = 2)  # Get max value and optimal choice
        idx = [i[2] for i in idx]
        
        # Extract policy function
        k′ = kgrid[idx]  
    
        # Convergence check
        dif = norm(k′ - kold, Inf);
        iter += 1
    end

    return vec(Vnew), vec(k′), iter
end

"""
    Exploiting monotonicity in VFI
"""

function vfi3(NGM::NeoclassicalGrowthModel, Vguess::Vector{Float64}, tol::Float64)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, u, idx_initial = NGM;

    # Initialize values for the iteration 
    dif = 1;
    iter = 0;
    Vold = copy(Vguess);
    Vnew = copy(Vold);
    k′ = Vector{Float64}(undef, n);  # policy function for capital
    k′old = copy(k′);  # initialize previous policy function for convergence check

    # vfi 
    while dif > tol
        k′old = copy(k′)  # Update previous policy function
        Vold = copy(Vnew);
    
        for i = 1:n 
           
            if i == 1
                idx_initial = 1; # start searching in the first point of the grid
            end
            vec = zeros(n);
            for j = idx_initial:n  # loop on k′
                # compute consumption 
                c = kgrid[i]^α + (1 - δ) * kgrid[i] - kgrid[j];  # assuming k′ lies on the same grid as k
                
                # compute utility 
                if c > 0
                    util = u(c);
                else
                    util = -Inf;
                end

                # vector to evaluate and choose optimal k′
                vec[j] = util + β * Vold[j];
            end

            # compute optimal choice for k′
            Vnew[i], idx = findmax(vec)
            k′[i] = kgrid[idx];
            idx_initial = idx;
        end

        # Stopping rule based on policy function convergence:
        dif = norm(k′ - k′old, Inf);  # Convergence based on the change in the policy function        
        #dif = norm(Vnew - Vold, Inf);
        iter += 1;

        # Print iteration and difference between k′ and k′old
        #println("Iteration $iter: Policy function difference = ", dif)
    end

    return Vnew, k′, iter
end

"""
    Exploiting monotonicity in VFI without for loops, just matrices
"""

function vfi3_matrix(NGM::NeoclassicalGrowthModel, Vguess::Vector{Float64}, tol::Float64)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, u, idx_initial = NGM;
# TO BE DONE
end

"""
    Function to compute VFI with Howard's Modified Policy Iteration (MPI) algorithm
"""
function vf_mpi(NGM::NeoclassicalGrowthModel,Vguess::Vector{Float64}, tol::Float64, m::Int)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, nbf_mpi, u = NGM;

    # Initialize values for the iteration 
    dif = 1;
    iter = 0;
    Vold = copy(Vguess);
    count_m = 0;
    Vnew = copy(Vold);
    k′ = Vector{Float64}(undef, n);  # policy function for capital
    k′old = copy(k′);  # initialize previous policy function for convergence check
    
    # vfi 
    while dif > tol
        k′old = copy(k′)  # Update previous policy function
        Vold = copy(Vnew);
        
        if iter == 0 || count_m == m  # in these cases I take the max #(I thought about including: || iter < nbf_mpi, but it makes the code slower).
            for i = 1:n 
                vec = Vector{Float64}(undef, n);
                
                for j = 1:n  # loop on k′
                    # compute consumption 
                    c = kgrid[i]^α + (1 - δ) * kgrid[i] - kgrid[j];  # assuming k′ lies on the same grid as k
                    
                    # compute utility 
                    if c > 0
                        util = u(c);
                    else
                        util = -Inf;
                    end

                    # vector to evaluate and choose optimal k′
                    vec[j] = util + β * Vold[j];
                end

                # compute optimal choice for k′
                Vnew[i], idx = findmax(vec)
                k′[i] = kgrid[idx];

                count_m = 0;
            end
            # Stopping rule based on policy function convergence:
            dif = norm(k′ - k′old, Inf);
        else # here we repeat the previous policy function and just updates the value function
            for i = 1:n 
                c = kgrid[i]^α + (1 - δ) * kgrid[i] - k′[i];
                # compute utility 
                if c > 0
                    util = u(c);
                else
                    util = -Inf;
                end
                # update value function
                Vnew[i] = util + β * Vold[findfirst(x -> x==k′[i],kgrid)];
            end
            count_m += 1;
        end 

        #= Note for future me: if you have a stopping rule based on value function convergence, 
            we can update dif here. However, it's based on policy function convergence, note that
            it has necessarily to be updated inside the if condition. Otherwise, in the second iteration
                the variable diff would be updated to precisely zero and the algorithm would stop.        
        =#
        # Stopping rule based on value function convergence:      
        #dif = norm(Vnew-Vold, Inf);
        iter += 1;

        # Print iteration and difference between Vnew and Vold
        # println("Iteration $iter: Policy function difference = ", dif)
    end

    return Vnew, k′, iter
end

"""
    Function to compute VFI with Howard's Modified Policy Iteration (MPI) algorithm without for loops, just matrices
"""
function vf_mpi_matrix(NGM::NeoclassicalGrowthModel,Vguess::Vector{Float64}, tol::Float64, m::Int)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, nbf_mpi, u = NGM;

    Vold = copy(Vguess)
    Vnew = copy(Vguess)
    k′ = similar(kgrid)  
    kold = similar(kgrid)
    idx = similar(kgrid)
    count_m = 0;
    dif = 1
    iter = 0
    
    C = kgrid.^α .+ (1 - δ) .* kgrid .- kgrid'  
    U = similar(C)
    U .= -Inf  # Initialize with -Inf
    mask = C .> 0
    U[mask] .= u.(C[mask])

    while dif > tol
        kold = copy(k′)  # Update previous policy function
        Vold = copy(Vnew)
        
        if iter == 0 || count_m == m # in these cases I take the max
            # Bellman update:
            Vmat = U .+ β .* Vold'
            Vnew, idx = findmax(Vmat,dims = 2)  # Get max value and optimal choice
            idx = [i[2] for i in idx]
            # Extract policy function
            k′ = kgrid[idx]  
            # Convergence check
            dif = norm(k′ - kold, Inf);
            count_m = 0;
        else # here we repeat the previous policy function and just updates the value function
            # update value function
            #Vnew = [U[i,idx[i]] + β * Vold[idx[i]] for i in 1:n]
            Vnew .= U[CartesianIndex.(1:n, idx)] .+ β .* Vold[idx]
            count_m += 1;
        end
 
        iter += 1
    end

    return vec(Vnew), vec(k′), iter
end

"""
    Function to solve VFI accelerating it with MQP error bounds
"""

function vf_mqp(NGM::NeoclassicalGrowthModel, Vguess::Vector{Float64}, tol::Float64, m::Int)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, u = NGM;

    # Initialize values for the iteration 
    dif = 1;
    iter = 0;
    Vold = copy(Vguess);
    Vnew = copy(Vold);
    k′ = Vector{Float64}(undef, n);  # policy function for capital
    k′old = copy(k′);  # initialize previous policy function for convergence check
    count_m = 0;
    c_upp = c_low = 0;

    # vfi 
    while dif > tol
        k′old = copy(k′)  # Update previous policy function
        Vold = copy(Vnew);
        for i = 1:n 
            vec = Vector{Float64}(undef, n);;
            
            for j = 1:n  # loop on k′
                # compute consumption 
                c = kgrid[i]^α + (1 - δ) * kgrid[i] - kgrid[j];  # assuming k′ lies on the same grid as k
                
                # compute utility 
                if c > 0
                    util = u(c);
                else
                    util = -Inf;
                end

                # vector to evaluate and choose optimal k′
                vec[j] = util + β * Vold[j];
            end

            # compute optimal choice for k′
            Vnew[i], idx = findmax(vec)
            k′[i] = kgrid[idx];
        end

        #compute c_lower and c_upper
        c_low = β/(1-β) * minimum(Vnew.-Vold)
        c_upp = β/(1-β) * maximum(Vnew.-Vold)

        if count_m == m 
            count_m = 0
        else 
            # update Vnew with the bounds:
            Vnew .= Vnew .+ (c_low+c_upp)/2
            count_m += 1
        end        

        # Stopping rule based on policy function convergence:
        dif = norm(k′ - k′old, Inf);  # Convergence based on the change in the policy function        
        #dif = c_upp-c_low;
        
        iter += 1;

        # Print iteration and difference between k′ and k′old
        #println("Iteration $iter: Policy function difference = ", dif)
    end

    return Vnew, k′, iter
end

"""
    Function to solve VFI accelerating it with MQP error bounds but without for loops (just matrices)
"""

function vf_mqp_matrix(NGM::NeoclassicalGrowthModel, Vguess::Vector{Float64}, tol::Float64, m::Int)::Tuple{Vector{Float64}, Vector{Float64}, Int64}
    @unpack α, β, δ, n, kgrid, u = NGM
    Vold = copy(Vguess)
    Vnew = copy(Vguess)
    k′ = similar(kgrid)  
    kold = similar(kgrid)
    dif = 1
    iter = 0
    count_m = 0
    c_upp = c_low = 0
    
    C = kgrid.^α .+ (1 - δ) .* kgrid .- kgrid'  
    U = similar(C)
    U .= -Inf  # Initialize with -Inf
    mask = C .> 0
    U[mask] .= u.(C[mask])

    while dif > tol
        kold = copy(k′)  # Update previous policy function
        Vold = copy(Vnew)
    
        # Bellman update:
        Vmat = U .+ β .* Vold'
        Vnew, idx = findmax(Vmat,dims = 2)  # Get max value and optimal choice
        idx = [i[2] for i in idx]
        
        # Extract policy function
        k′ = kgrid[idx]  

        #compute c_lower and c_upper
        c_low = β/(1-β) * minimum(Vnew.-Vold)
        c_upp = β/(1-β) * maximum(Vnew.-Vold)

        if count_m == m 
            count_m = 0
        else 
            # update Vnew with the bounds:
            Vnew .= Vnew .+ (c_low+c_upp)/2
            count_m += 1
        end                
    
        # Convergence check
        dif = norm(k′ - kold, Inf);
        iter += 1
    end

    return vec(Vnew), vec(k′), iter
end


"""
    Function to compute VFI with Howard's Modified Policy Iteration (MPI) algorithm without for loops, just matrices
    the difference between this and the previous MPI function is that this updates the last value function (used in (c.i))
"""
function vf_mpi_matrix2(NGM::NeoclassicalGrowthModel,Vguess::Vector{Float64}, tol::Float64, m::Int)::Tuple{Vector{Float64}, Vector{Float64}, Int64, Vector{Float64}}
    @unpack α, β, δ, n, kgrid, nbf_mpi, u = NGM;

    Vold = copy(Vguess)
    Vnew = copy(Vguess)
    k′ = similar(kgrid)  
    kold = similar(kgrid)
    idx = similar(kgrid)
    count_m = 0;
    dif = 1
    iter = 0
    
    C = kgrid.^α .+ (1 - δ) .* kgrid .- kgrid'  
    U = similar(C)
    U .= -Inf  # Initialize with -Inf
    mask = C .> 0
    U[mask] .= u.(C[mask])

    while dif > tol
        kold = copy(k′)  # Update previous policy function
        Vold = copy(Vnew)
        
        if iter == 0 || count_m == m # in these cases I take the max
            # Bellman update:
            Vmat = U .+ β .* Vold'
            Vnew, idx = findmax(Vmat,dims = 2)  # Get max value and optimal choice
            idx = [i[2] for i in idx]
            # Extract policy function
            k′ = kgrid[idx]  
            # Convergence check
            dif = norm(k′ - kold, Inf);
            count_m = 0;
        else # here we repeat the previous policy function and just updates the value function
            # update value function
            #Vnew = [U[i,idx[i]] + β * Vold[idx[i]] for i in 1:n]
            Vnew .= U[CartesianIndex.(1:n, idx)] .+ β .* Vold[idx]
            count_m += 1;
        end
 
        iter += 1
    end

    return vec(Vnew), vec(k′), iter, vec(Vold)
end

"""
    Function to compute VFI with Howard's Modified Policy Iteration (MPI) algorithm without for loops, just matrices
    the difference between this and the previous MPI function is that this uses a different stopping rule
"""
function vf_mpi_matrix3(NGM::NeoclassicalGrowthModel,Vguess::Vector{Float64}, tol::Float64, m::Int)::Tuple{Vector{Float64}, Vector{Float64}, Int64, Vector{Float64}}
    @unpack α, β, δ, n, kgrid, nbf_mpi, u = NGM;

    Vold = copy(Vguess)
    Vnew = copy(Vguess)
    Vupdated = copy(Vguess)
    k′ = similar(kgrid)  
    kold = similar(kgrid)
    idx = similar(kgrid)
    count_m = 0;
    c_low = c_upp = 0;
    dif = 1
    iter = 0
    
    C = kgrid.^α .+ (1 - δ) .* kgrid .- kgrid'  
    U = similar(C)
    U .= -Inf  # Initialize with -Inf
    mask = C .> 0
    U[mask] .= u.(C[mask])

    while dif > tol
        kold = copy(k′)  # Update previous policy function
        Vold = copy(Vnew)
        
        if iter == 0 || count_m == m # in these cases I take the max
            # Bellman update:
            Vmat = U .+ β .* Vold'
            Vnew, idx = findmax(Vmat,dims = 2)  # Get max value and optimal choice
            idx = [i[2] for i in idx]
            # Extract policy function
            k′ = kgrid[idx] 
            #compute c_lower and c_upper
            c_low = β/(1-β) * minimum(Vnew.-Vold)
            c_upp = β/(1-β) * maximum(Vnew.-Vold) 
            #update dif
            dif = c_upp-c_low
            count_m = 0;
        else # here we repeat the previous policy function and just updates the value function
            # update value function
            #Vnew = [U[i,idx[i]] + β * Vold[idx[i]] for i in 1:n]
            Vnew .= U[CartesianIndex.(1:n, idx)] .+ β .* Vold[idx]
            count_m += 1;
        end

        # update Vnew with the bounds:
        Vupdated .= Vnew .+ (c_low+c_upp)/2

        iter += 1
    end

    return vec(Vnew), vec(k′), iter, vec(Vupdated)
end

