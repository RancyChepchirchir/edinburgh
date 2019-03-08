library(ggplot2)
library(data.table)
library(reshape2)

# main_dir <- '/afs/inf.ed.ac.uk/user/s16/s1667278/Desktop/rl-cw1'
main_dir <- '/Users/sipola/Google Drive/education/coursework/graduate/edinburgh/rl/rl-cw1'
run_dir <- file.path(main_dir, 'run')
create_reward_path <- function(run_dir, txt) {
  file.path(run_dir, txt, 'total_reward.txt')
}

random <- fread(create_reward_path(run_dir, 'random_agent'))
accel <- fread(create_reward_path(run_dir, 'accelerate_agent'))
qagent <- fread(create_reward_path(run_dir, 'f4_s7/saved'))
qagent.close <- fread(create_reward_path(run_dir, 'f3_s7/saved'))
qagent.far <- fread(create_reward_path(run_dir, 'f6_s7/saved'))

setnames(random, c('V1', 'V2'), c('Episode', 'random_agent'))
setnames(accel, c('V1', 'V2'), c('Episode', 'accelerate_agent'))
setnames(qagent, c('V1', 'V2'), c('Episode', 'q_agent'))
setnames(qagent.close, c('V1', 'V2'), c('Episode', 'q_agent (close vision)'))
setnames(qagent.far, c('V1', 'V2'), c('Episode', 'q_agent (far vision)'))

X <- merge(random, accel, by = 'Episode')
X <- merge(X, qagent, by = 'Episode')
X <- merge(X, qagent.close, by = 'Episode')
X <- merge(X, qagent.far, by = 'Episode')

X <- data.table(melt(X, id = 'Episode', variable.name = 'Agent'))

agents.graph1 <- c('random_agent', 'accelerate_agent', 'q_agent')
agents.graph2 <- c('q_agent', 'q_agent (close vision)', 'q_agent (far vision)')
ggplot(X[Agent %in% c(agents.graph1)], aes(Episode, value, color = Agent)) + 
  geom_line() +
  ggtitle('Reward by agent type') +
  ylab('Total reward')

ggplot(X[Agent %in% c(agents.graph2)], aes(Episode, value, color = Agent)) + 
  geom_line() +
  ggtitle('Reward by vision distance') +
  ylab('Total reward')

for (agent in X[, unique(Agent)]) {
  X.agent <- X[Agent == agent]
  agent.mean <- signif(X.agent[, mean(value)], 2)
  agent.sd <- signif(X.agent[, sd(value)], 2)
  g <- ggplot(X.agent, aes(value)) +
    geom_bar() +
    ggtitle(paste0('Distribution of rewards for ', agent, '\nmean = ', agent.mean, ', sd = ', agent.sd)) +
    xlab('Total reward') +
    ylab('Count')
  ggsave(file.path(run_dir, paste0('hist_', agent, '.pdf')), g, width = 8, height = 3, units = 'in')
}
ggplot(X[Agent == 'q_agent'], aes(value)) +
  geom_bar() +
  ggtitle('Distribution of rewards\nmean = , sd = ')
ggplot(X[Agent == 'random_agent'], aes(value)) + geom_bar()