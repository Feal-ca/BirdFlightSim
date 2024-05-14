mi_file <-
  read.csv("/home/georgiou/Documents/Biofis/animalTrait.csv")

columns <-
  c("order",
    "family",
    "genus",
    "species",
    "body.mass",
    "metabolic.rate",
    "inTextReference")  # Vector of column names to remove
birds <- subset(mi_file, class == "Aves")
mammals<- subset(mi_file, class == "Mammalia")
mammals <- mammals[, columns, drop = FALSE]

birds <- birds[, columns, drop = FALSE]
birds_filtered <-
  subset(birds,!is.na(metabolic.rate))
birds_filtered <-
  subset(birds_filtered,!is.na(body.mass))
birds_filtered <-
  subset(birds_filtered,(body.mass)>0.1)

laws <- subset(mi_file, inTextReference == "Lasiewski & Dawson, 1967
")


passerins <- subset(birds, order == "Passeriformes")
non_passerins <- subset(birds, order != "Passeriformes")

mammals_filtered <-
  subset(mammals,!is.na(metabolic.rate))
mammals_filtered <-
  subset(mammals_filtered,!is.na(body.mass))
# mammals_filtered <-
#   subset(mammals_filtered,(body.mass)>0.1)

non_passerins_filtered <-
  subset(non_passerins,!is.na(metabolic.rate))
non_passerins_filtered <-
  subset(non_passerins_filtered,!is.na(body.mass))
# non_passerins_filtered <- subset(non_passerins_filtered,(body.mass)>0.1)

passerins_filtered <- subset(passerins,!is.na(metabolic.rate))
passerins_filtered <- subset(passerins_filtered,!is.na(body.mass))
# passerins_filtered <- subset(passerins_filtered,(body.mass)>0.1)

passerins_filtered$metabolic_rate_body_mass_ratio <-
  passerins_filtered$metabolic.rate / passerins_filtered$body.mass
non_passerins_filtered$metabolic_rate_body_mass_ratio <-
  non_passerins_filtered$metabolic.rate / non_passerins_filtered$body.mass

regr_pas_bmmr <-
  lm(log(metabolic_rate_body_mass_ratio) ~ log(body.mass), data = passerins_filtered)
regr_non_pas_bmmr <-
  lm(log(metabolic_rate_body_mass_ratio) ~ log(body.mass), data = non_passerins_filtered)
regr_pas <-
  lm(log(metabolic.rate) ~ log(body.mass), data = passerins_filtered)
regr_mam <-
  lm(log(metabolic.rate) ~ log(body.mass), data = mammals_filtered)
regr_bird <-
  lm(log(metabolic.rate) ~ log(body.mass), data = birds_filtered)
regr_non_pas <-
  lm(log(metabolic.rate) ~ log(body.mass), data = non_passerins_filtered)

plot(
  log(mammals_filtered$body.mass),
  log(mammals_filtered$metabolic.rate),
  xlab = "Log Body Mass",
  ylab = "Log Metabolic Rate",
  main = "Mammals",
  col = "red",
  pch = 20
)
plot(
  log(birds$body.mass),
  log(birds$metabolic.rate),
  xlab = "Body Mass",
  ylab = "Metabolic Rate",
  main = "Birds",
  col = "black",
  pch = 20
)
Add the regression line for passerine birds

 # Plot the points for passerine birds
 points(
   log(passerins_filtered$body.mass),
   log(passerins_filtered$metabolic.rate),
   xlab = "Body Mass [kg]",
   ylab = "Metabolic Rate [W]",
   main = "Passerines",
   col = "blue",
   pch = 20
)

# # Add the regression line for passerine birds
 abline(regr_pas, col = "blue")
 abline(regr_mam, col = "red")
 abline(regr_non_pas, col = "green")
 
Plot the points for non-passerine birds
points(
  log(non_passerins_filtered$body.mass),
  log(non_passerins_filtered$metabolic.rate),
  col = "green",
  pch = 20
)


legend(
  "topleft",
  legend = c("Passerines", "Non-Passerines", "Mammals"),
  col = c("blue", "green", "red"),
  pch = 16
)

