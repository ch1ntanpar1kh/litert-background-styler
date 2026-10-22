pluginManagement {
  repositories {
    google()
    mavenCentral()
    gradlePluginPortal()
  }
}

dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    mavenCentral()
    // Early-access LiteRT in Google Play services SDK (play-services-litert 17.0.0-eap01).
    // Not checked in: point -PgmsLitertEapRepo=<dir> at the unzipped EAP Maven repo, or put it at
    // ~/gms-litert-eap-maven.
    maven {
      url =
        uri(
          providers.gradleProperty("gmsLitertEapRepo").orNull
            ?: "${System.getProperty("user.home")}/gms-litert-eap-maven"
        )
      content { includeGroup("com.google.android.gms") }
    }
  }
}

rootProject.name = "BackgroundStyler"

include(":app")
