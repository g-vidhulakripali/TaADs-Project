import { bootstrapApplication } from '@angular/platform-browser';
import { provideRouter, Routes } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { importProvidersFrom } from '@angular/core';
import { SearchPageComponent } from './app/search-page/search-page.component';
import { AppComponent } from './app/app.component';
import { FormsModule } from '@angular/forms';

// Define routes
const routes: Routes = [
  { path: 'search', component: SearchPageComponent },
  { path: '', redirectTo: '/search', pathMatch: 'full' }
];

// Bootstrap the root component with providers
bootstrapApplication(AppComponent, {
  providers: [
    provideRouter(routes),
    provideHttpClient(),
    importProvidersFrom(FormsModule),
  ]
}).catch(err => console.error(err));
