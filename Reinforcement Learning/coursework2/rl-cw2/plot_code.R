library(ggplot2)
library(data.table)
library(reshape2)

# main_dir <- '/afs/inf.ed.ac.uk/user/s16/s1667278/Desktop/rl-cw2'
main_dir <- '/Users/sipola/Google Drive/education/coursework/graduate/edinburgh/rl/rl-cw2/rl-cw2'
run_dir <- file.path(main_dir, 'run')
create_reward_path <- function(run_dir, txt) {
  file.path(run_dir, txt, 'total_reward.txt')
}

qagent <- fread(create_reward_path('/afs/inf.ed.ac.uk/user/s16/s1667278/Desktop/rl-cw1/run', 'f4_s7'))
linear <- fread(create_reward_path(run_dir, 'initial/saved'))

setnames(qagent, c('V1', 'V2'), c('Episode', 'q_agent'))
setnames(linear, c('V1', 'V2'), c('Episode', 'linear_approx'))

X <- merge(qagent, linear, by = 'Episode')
X <- data.table(melt(X, id = 'Episode', variable.name = 'Agent'))

agents.graph1 <- c('q_agent', 'linear_approx')
g <- ggplot(X[Agent %in% c(agents.graph1)], aes(Episode, value, color=Agent)) + 
  geom_line(alpha=0.5) +
  geom_smooth(se = FALSE, span=1/10, lwd=1) + 
  ggtitle('Reward by agent type') +
  ylab('Total reward')
ggsave(file.path(run_dir, 'comparisons.pdf'), g, width = 8, height = 4, units = 'in')

# Plot features by episode.
# ff <- fread('/afs/inf.ed.ac.uk/user/s16/s1667278/Desktop/rl-cw2/run/initial/weights_all.csv')
ff <- fread('/Users/sipola/Google Drive/education/coursework/graduate/edinburgh/rl/rl-cw2/rl-cw2/run/initial/weights_all.csv')
ff[, Episode := .I]
ff <- melt(ff, id.vars = 'Episode', variable.name = 'Feature')
g <- ggplot(ff, aes(Episode, value, color=Feature, linetype=Feature)) + 
  geom_line(aes(shape = Feature)) +
  ggtitle('Feature weights over episodes') +
  ylab('Feature weight')
ggsave(file.path(run_dir, 'features_lin_approx.pdf'), g, width = 8, height = 4, units = 'in')


# ggplot(X[Agent %in% c(agents.graph2)], aes(Episode, value, color = Agent)) + 
#   geom_line() +
#   ggtitle('Reward by vision distance') +
#   ylab('Total reward')

# for (agent in X[, unique(Agent)]) {
#   X.agent <- X[Agent == agent]
#   agent.mean <- signif(X.agent[, mean(value)], 2)
#   agent.sd <- signif(X.agent[, sd(value)], 2)
#   g <- ggplot(X.agent, aes(value)) +
#     geom_bar() +
#     ggtitle(paste0('Distribution of rewards for ', agent, '\nmean = ', agent.mean, ', sd = ', agent.sd)) +
#     xlab('Total reward') +
#     ylab('Count')
#   ggsave(file.path(run_dir, paste0('hist_', agent, '.pdf')), g, width = 8, height = 3, units = 'in')
# }
# ggplot(X[Agent == 'q_agent'], aes(value)) +
#   geom_bar() +
#   ggtitle('Distribution of rewards\nmean = , sd = ')
# ggplot(X[Agent == 'random_agent'], aes(value)) + geom_bar()