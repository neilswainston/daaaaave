% kieran: 26 apr 12

clc

rsquared = @(f,y)(1 - sum((y-f).^2)/sum((y-mean(y)).^2));

%% example

model_name      = 'example.xml';
gene_to_scale   = 'rA';
flux_to_scale   = 'rA';

% #1

[reaction_name,experimental,p_gene_exp,p_standard_fba,p_standard_fba_best,p_gimme,p_shlomi] = ...
    analysis(model_name, 'genedata_example_1.txt','experimental_fluxes_example_1.txt',gene_to_scale,flux_to_scale);

% display
fprintf('%s\t%s\t%s\t%s\t%s\t%s\t%s\n','Reaction','Example 1','Daaaaave','FBA','Fitted FBA','Gimme','Shlomi');
for k = 1:size(reaction_name,1)
    fprintf('%s\t%g\t%g\t%g\t%g\t%g\t%g\n',reaction_name{k,1},experimental(k),p_gene_exp(k),p_standard_fba(k),p_standard_fba_best(k),p_gimme(k),p_shlomi(k));
end
fprintf('%s\t%g\t%g\t%g\t%g\t%g\t%g\n\n','R2',1,rsquared(p_gene_exp,experimental),rsquared(p_standard_fba,experimental),rsquared(p_standard_fba_best,experimental),rsquared(p_gimme,experimental),rsquared(p_shlomi,experimental));

% #2

[reaction_name,experimental,p_gene_exp,p_standard_fba,p_standard_fba_best,p_gimme,p_shlomi] = ...
    analysis(model_name, 'genedata_example_2.txt','experimental_fluxes_example_2.txt',gene_to_scale,flux_to_scale);

% display
fprintf('%s\t%s\t%s\t%s\t%s\t%s\t%s\n','Reaction','Example 2','Daaaaave','FBA','Fitted FBA','Gimme','Shlomi');
for k = 1:size(reaction_name,1)
    fprintf('%s\t%g\t%g\t%g\t%g\t%g\t%g\n',reaction_name{k,1},experimental(k),p_gene_exp(k),p_standard_fba(k),p_standard_fba_best(k),p_gimme(k),p_shlomi(k));
end
fprintf('%s\t%g\t%g\t%g\t%g\t%g\t%g\n\n','R2',1,rsquared(p_gene_exp,experimental),rsquared(p_standard_fba,experimental),rsquared(p_standard_fba_best,experimental),rsquared(p_gimme,experimental),rsquared(p_shlomi,experimental));

%% real

model_name      = 'yeast_5.21_MCISB.xml';
gene_to_scale   = 'glucose transport';
flux_to_scale   = 'D-glucose exchange';

% 75%

[reaction_name,experimental,p_gene_exp,p_standard_fba,p_standard_fba_best,p_gimme,p_shlomi] = ...
    analysis(model_name, 'genedata_75.txt','experimental_fluxes_75.txt',gene_to_scale,flux_to_scale);

% display
fprintf('%s\t%s\t%s\t%s\t%s\t%s\t%s\n','Reaction','Experimental 75%','Daaaaave','FBA','Fitted FBA','Gimme','Shlomi');
for k = 1:size(reaction_name,1)
    fprintf('%s\t%g\t%g\t%g\t%g\t%g\t%g\n',reaction_name{k,1},experimental(k),p_gene_exp(k),p_standard_fba(k),p_standard_fba_best(k),p_gimme(k),p_shlomi(k));
end
fprintf('%s\t%g\t%g\t%g\t%g\t%g\t%g\n\n','R2',1,rsquared(p_gene_exp,experimental),rsquared(p_standard_fba,experimental),rsquared(p_standard_fba_best,experimental),rsquared(p_gimme,experimental),rsquared(p_shlomi,experimental));

% 85%

[reaction_name,experimental,p_gene_exp,p_standard_fba,p_standard_fba_best,p_gimme,p_shlomi] = ...
    analysis(model_name, 'genedata_85.txt','experimental_fluxes_85.txt',gene_to_scale,flux_to_scale);

% display
fprintf('%s\t%s\t%s\t%s\t%s\t%s\t%s\n','Reaction','Experimental 85%','Daaaaave','FBA','Fitted FBA','Gimme','Shlomi');
for k = 1:size(reaction_name,1)
    fprintf('%s\t%g\t%g\t%g\t%g\t%g\t%g\n',reaction_name{k,1},experimental(k),p_gene_exp(k),p_standard_fba(k),p_standard_fba_best(k),p_gimme(k),p_shlomi(k));
end
fprintf('%s\t%g\t%g\t%g\t%g\t%g\t%g\n\n','R2',1,rsquared(p_gene_exp,experimental),rsquared(p_standard_fba,experimental),rsquared(p_standard_fba_best,experimental),rsquared(p_gimme,experimental),rsquared(p_shlomi,experimental));
