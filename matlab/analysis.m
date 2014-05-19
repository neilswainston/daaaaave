% kieran: 26 apr 12

function [reaction_name,experimental,p_gene_exp,p_standard_fba,p_standard_fba_best,...
    p_gimme,p_shlomi] = ...
    analysis(model_filename,genedata_filename,experimental_fluxes_filename,...
    gene_to_scale,flux_to_scale)

% load model
model       = readCbModel(model_filename);

% load transcript data
genedata	= importdata(genedata_filename);
genenames	= genedata.textdata(:,1);
genenames(1)= [];
gene_exp	= genedata.data(:,1);
gene_exp_sd	= genedata.data(:,2);

% map gene weighting to reaction weighting
[rxn_exp,rxn_exp_sd] = geneToReaction(model,genenames,gene_exp,gene_exp_sd);
% sds 0 -> small
rxn_exp_sd(rxn_exp_sd == 0) = min(rxn_exp_sd(rxn_exp_sd>0))/2;

% scale by uptake reaction
uptake              = find(strcmp(gene_to_scale,model.rxnNames));
rxn_exp_sd          = rxn_exp_sd/rxn_exp(uptake);
rxn_exp             = rxn_exp/rxn_exp(uptake);
model.lb(uptake)	= 1;
model.ub(uptake)	= 1;

% Gene expression constraint FBA
v_gene_exp      = dataToFlux(model,rxn_exp,rxn_exp_sd);

% Standard FBA
solution        = optimizeCbModel(model,[],'one');
v_standard_fba	= solution.x;
fOpt            = solution.f;

% "required metabolic functionalities" (=growth) set to 90% of maximum
model.lb(model.c == 1)	= 0.9*fOpt;
% gimme
v_gimme         = zeros(size(model.rxns));
try 
    v_gimme     = gimme(model,rxn_exp);
catch %#ok<CTCH>
    disp('gimme algorithm unsuccessful');
end

% schlomi
v_shlomi       	= zeros(size(model.rxns));
try 
    v_shlomi    = shlomi(model,rxn_exp);
catch %#ok<CTCH>
    disp('shlomi algorithm unsuccessful');
end

% compare
experimental_fluxes = importdata(experimental_fluxes_filename);

reaction_name   = experimental_fluxes.textdata;
experimental	= zeros(size(experimental_fluxes.textdata,1),1);
p_gene_exp      = zeros(size(experimental_fluxes.textdata,1),1);
p_standard_fba	= p_gene_exp;
p_gimme         = p_gene_exp;
p_shlomi     	= p_gene_exp;

flux = strcmp(flux_to_scale,reaction_name);
flux = experimental_fluxes.data(flux,1);

for k = 1:size(experimental_fluxes.textdata,1)
    j = find(strcmp(reaction_name{k},model.rxnNames));
    experimental(k)     = experimental_fluxes.data(k,1);
    p_gene_exp(k)       = flux*abs(v_gene_exp(j));
    p_standard_fba(k)	= flux*abs(v_standard_fba(j));
    p_gimme(k)          = flux*abs(v_gimme(j));
    p_shlomi(k)     	= flux*abs(v_shlomi(j));
end

% remove small entries
p_gene_exp(abs(p_gene_exp)<1e-6)            = 0;
p_standard_fba(abs(p_standard_fba)<1e-6)	= 0;
p_gimme(abs(p_gimme)<1e-6)                  = 0;
p_shlomi(abs(p_shlomi)<1e-6)                = 0;

% find best fit from standard FBA solution
% ... overkill for this problem, but reuses existing method
model.lb(model.c == 1) = fOpt;
data = nan(size(model.rxns));
data(uptake) = 1;
for k = 1:size(experimental_fluxes.textdata,1)
    j = find(strcmp(experimental_fluxes.textdata{k,1},model.rxnNames));
    data(j) = experimental_fluxes.data(k,1)/flux; %#ok<FNDSB>
end
v_standard_fba_best = dataToFlux(model,data,data);
p_standard_fba_best	= zeros(size(experimental_fluxes.textdata,1),1);
for k = 1:size(experimental_fluxes.textdata,1)
    j = find(strcmp(experimental_fluxes.textdata{k,1},model.rxnNames));
    p_standard_fba_best(k) = flux*abs(v_standard_fba_best(j)); %#ok<FNDSB>
end
p_standard_fba_best(abs(p_standard_fba_best)<1e-6)	= 0;

function [r,r_sd] = geneToReaction(m,g,t,t_sd)

% kieran: 16 sep 11

r       = zeros(size(m.rxns));
r_sd    = zeros(size(m.rxns));

for k = 1:length(g)
    g{k} = strrep(g{k},'-','_');
end

for k = 1:length(m.rxns)
    ga = m.grRules{k};
    ga = strrep(ga,'-','_');
    w = regexp(ga,'\<\w*\>','match'); 
    w = setdiff(w,{'and','or'});
    for kk = 1:length(w)
		j = find(strcmp(w{kk},g));
        n = t(j);
        n_sd = t_sd(j);        
        ga = regexprep(ga,['\<',w{kk},'\>'],[num2str(n),'�',num2str(n_sd)]);
    end
    [n,n_sd] = addGeneData(ga);
    r(k) = n;
    r_sd(k) = n_sd;
end

function [n,n_sd] = addGeneData(g)

% kieran: 22 july 11

n = nan;
n_sd = nan;

ApmB = '[0-9\.]+�[0-9\.]+';

if ~isempty(g)
    while isnan(n)
        
        try 
            match_expr      = '([0-9\.])+�([0-9\.]+)';
            g_av            = regexprep(g,match_expr,'$1');
            g_sd            = regexprep(g,match_expr,'$2');
            n               = eval(g_av);
            n_sd            = eval(g_sd);
            
        catch %#ok<CTCH>
            
            % replace brackets
            match_expr      = ['\((',ApmB,')\)'];
            replace_expr    = '$1';
            g = regexprep(g,match_expr,replace_expr);
            
            % replace and
            match_expr      = ['(',ApmB,') and (',ApmB,')'];
            replace_expr    = '${AandB($1,$2)}';
            g = regexprep(g,match_expr,replace_expr,'once');
            
            % replace or
            match_expr      = ['(',ApmB,') or (',ApmB,')'];
            replace_expr    = '${AorB($1,$2)}';
            g = regexprep(g,match_expr,replace_expr,'once');
           
        end
    end
end

function v_sol = dataToFlux(m,r,r_sd)

% kieran: 21 sep 11

rev = false(size(m.rxns));

nR_old = 0;

% m.lb(~ismember(m.lb,[-inf,0,inf,-1000,1000])) = -inf;
% m.ub(~ismember(m.ub,[-inf,0,inf,-1000,1000])) = inf;

v_sol = zeros(size(m.rxns));

while sum(~m.rev) > nR_old
    
    nR_old = sum(~m.rev);
    
    % 1. fit to data
    
    N = m.S;    
    L = m.lb;
    U = m.ub;
    f = zeros(size(m.rxns))';
    b = zeros(size(m.mets));
    
    for k = 1:length(m.rxns)
        d = r(k);
        s = r_sd(k);
        
        if ~m.rev(k) && ~isnan(d) && s>0
            [s1,s2] = size(N);
            N(s1+1,k) = 1; N(s1+1,s2+1) = -1; N(s1+1,s2+2) = 1;
            L(s2+1) = 0; L(s2+2) = 0;
            U(s2+1) = inf; U(s2+2) = inf;
            b(s1+1) = d;
            f(s2+1) = - 1/s;
            f(s2+2) = - 1/s;
        end
    end
    
    [v,fOpt,conv] = easyLP(f,N,b,L,U);
    
    if conv
        v_sol = v(1:length(m.rxns));
        for k = 1:length(m.rxns)
            if rev(k), v_sol(k) = - v_sol(k); end
        end
        
        % 2. run FVA
        
        N = [N; f]; %#ok<AGROW>
        b = [b(:); fOpt];
        
        for k = 1:length(m.rxns)
            
            if m.rev(k)
                
                f = zeros(size(L));
                f(k) = -1;
                
                [~,fOpt,conv] = easyLP(f,N,b,L,U);
                
                if conv && (-fOpt >= 0) % irreversibly forward
                    
                    m.lb(k) = max(m.lb(k),0);
                    m.rev(k) = 0;
                    
                else
                    f(k) = 1;
                    [~,fOpt,conv] = easyLP(f,N,b,L,U);
                    
                    if conv && abs(fOpt)<=0 % irreversibly backward
                        
                        m.S(:,k) = - m.S(:,k);
                        
                        m.ub(k) = - m.ub(k);
                        m.lb(k) = - m.lb(k);
                        
                        ub = m.ub(k);
                        m.ub(k) = m.lb(k);
                        m.lb(k) = ub;
                        
                        m.lb(k) = max(m.lb(k),0);
                        m.rev(k) = 0;
                        
                        rev(k) = ~rev(k);
                        
                    end
                end
            end
        end
    end
end

function [v,fOpt,conv] = easyLP(f,a,b,vlb,vub)
%
%easyLP
%
% solves the linear programming problem: 
%   max f'x subject to 
%   a x = b
%   vlb <= x <= vub. 
%
% Usage: [v,fOpt,conv] = easyLP(f,a,b,vlb,vub)
%
%   f           objective coefficient vector
%   a           LHS matrix
%   b           RHS vector
%   vlb         lower bound
%   vub         upper bound
%
%   v           solution vector
%   fOpt        objective value
%   conv        convergence of algorithm [0/1]
%
% the function is a wrapper for on the "solveCobraLP" script provided with
% the COBRA (COnstraint-Based Reconstruction and Analysis) toolbox 
% http://opencobra.s.net/
%
%kieran, 20 april 2010

% matlab can crash if inputs nan
if any(isnan(f))||any(any(isnan(a)))||any(isnan(b))...
        ||any(isnan(vlb))||any(isnan(vub))  
    error('nan inputs not allowed');
end

% initialize
v = zeros(size(vlb));
v = v(:);
f = full(f(:));
vlb = vlb(:);
vub = vub(:);

% remove any tight contstraints as some solvers require volume > 0
j1 = (vlb ~= vub); 
j2 = (vlb == vub);
v(j2) = vlb(j2);
b = b(:) - a*v;
a(:,j2) = []; 
vlb(j2) = []; 
vub(j2) = [];
f0 = f;
f(j2) = [];
fOpt = nan;

% solve
csense(1:length(b)) = 'E';

solution = solveCobraLP(...
    struct('A',a,'b',b,'c',f,'lb',vlb,'ub',vub,'osense',-1,'csense',csense));

% define outputs
conv = solution.stat == 1;

if conv
    v0 = solution.full;
    v(j1) = v0;
    fOpt = f0'*v;
end

function v_sol = gimme(model,gene_exp)

% cutoff at lower quartile
cutoff                  = quantile(gene_exp(~isnan(gene_exp)),0.25);

% set up big matrices
model.c = zeros(size(model.c));

for k = 1:length(gene_exp)
    if gene_exp(k) < cutoff
        c = cutoff - gene_exp(k);
        [n1,n2] = size(model.S);
        % v = v+ - v-;
        model.S(n1+1,k)    = 1; model.b(n1+1) = 0;
        model.S(n1+1,n2+1) = -1; model.lb(n2+1) = 0; model.ub(n2+1) = inf; model.c(n2+1) = -c;
        model.S(n1+1,n2+2) = 1; model.lb(n2+2) = 0; model.ub(n2+2) = inf; model.c(n2+2) = -c;
    end
end

solution        = optimizeCbModel(model,[],'one');
v_sol = solution.x(1:length(model.rxns));

function v_sol = shlomi(model,rxn_exp)

% cutoffs at lower and upper quantiles
ExpressedRxns = model.rxns(rxn_exp >= quantile(rxn_exp(~isnan(rxn_exp)),0.75));
UnExpressedRxns = model.rxns(rxn_exp <= quantile(rxn_exp(~isnan(rxn_exp)),0.25));

% below copied from createTissueSpecificModel.m, part of the Cobra toolbox
% http://opencobra.sf.net/

RHindex = findRxnIDs(model,ExpressedRxns);
RLindex = findRxnIDs(model,UnExpressedRxns);

S = model.S;
lb = model.lb;
ub = model.ub;
eps = 1;

% Creating A matrix
A = sparse(size(S,1)+2*length(RHindex)+2*length(RLindex),size(S,2)+2*length(RHindex)+length(RLindex));
[m,n,s] = find(S);
for i = 1:length(m)
    A(m(i),n(i)) = s(i); %#ok<SPRIX>
end

for i = 1:length(RHindex)
    A(i+size(S,1),RHindex(i)) = 1; %#ok<SPRIX>
    A(i+size(S,1),i+size(S,2)) = lb(RHindex(i)) - eps; %#ok<SPRIX>
    A(i+size(S,1)+length(RHindex),RHindex(i)) = 1; %#ok<SPRIX>
    A(i+size(S,1)+length(RHindex),i+size(S,2)+length(RHindex)+length(RLindex)) = ub(RHindex(i)) + eps; %#ok<SPRIX>
end

for i = 1:length(RLindex)
    A(i+size(S,1)+2*length(RHindex),RLindex(i)) = 1; %#ok<SPRIX>
    A(i+size(S,1)+2*length(RHindex),i+size(S,2)+length(RHindex)) = lb(RLindex(i)); %#ok<SPRIX>
    A(i+size(S,1)+2*length(RHindex)+length(RLindex),RLindex(i)) = 1; %#ok<SPRIX>
    A(i+size(S,1)+2*length(RHindex)+length(RLindex),i+size(S,2)+length(RHindex)) = ub(RLindex(i)); %#ok<SPRIX>
end

% Creating csense
csense1(1:size(S,1)) = 'E';
csense2(1:length(RHindex)) = 'G';
csense3(1:length(RHindex)) = 'L';
csense4(1:length(RLindex)) = 'G';
csense5(1:length(RLindex)) = 'L';
csense = [csense1 csense2 csense3 csense4 csense5];

% Creating lb and ub
lb_y = zeros(2*length(RHindex)+length(RLindex),1);
ub_y = ones(2*length(RHindex)+length(RLindex),1);
lb = [lb;lb_y];
ub = [ub;ub_y];

% Creating c
c_v = zeros(size(S,2),1);
c_y = ones(2*length(RHindex)+length(RLindex),1);
c = [c_v;c_y];

% Creating b
b_s = zeros(size(S,1),1);
lb_rh = lb(RHindex);
ub_rh = ub(RHindex);
lb_rl = lb(RLindex);
ub_rl = ub(RLindex);
b = [b_s;lb_rh;ub_rh;lb_rl;ub_rl];

% Creating vartype
vartype1(1:size(S,2),1) = 'C';
vartype2(1:2*length(RHindex)+length(RLindex),1) = 'B';
vartype = [vartype1;vartype2];

MILPproblem.A = A;
MILPproblem.b = b;
MILPproblem.c = c;
MILPproblem.lb = lb;
MILPproblem.ub = ub;
MILPproblem.csense = csense;
MILPproblem.vartype = vartype;
MILPproblem.osense = -1;
MILPproblem.x0 = [];

solution = solveCobraMILP(MILPproblem);

x = solution.cont;
for i = 1:length(x)
    if abs(x(i)) < 1e-6
        x(i,1) = 0;
    end
end

v_sol = x;
